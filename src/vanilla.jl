"""
MCTS solver type

Fields:

    n_iterations::Int64
        Number of iterations during each action() call.
        default: 100

    max_time::Float64
        Maximum amount of CPU time spent iterating through simulations.
        default: Inf

    depth::Int64:
        Maximum rollout horizon and tree depth.
        default: 10

    exploration_constant::Float64:
        Specifies how much the solver should explore.
        In the UCB equation, Q + c*sqrt(log(t/N)), c is the exploration constant.
        default: 1.0

    rng::AbstractRNG:
        Random number generator

    estimate_value::Any (rollout policy)
        Function, object, or number used to estimate the value at the leaf nodes.
        If this is a function `f`, `f(mdp, s, depth)` will be called to estimate the value.
        If this is an object `o`, `estimate_value(o, mdp, s, depth)` will be called.
        If this is a number, the value will be set to that number
        default: RolloutEstimator(RandomSolver(rng))

    init_Q::Any
        Function, object, or number used to set the initial Q(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_Q(o, mdp, s, a)` will be called.
        If this is a number, Q will be set to that number
        default: 0.0

    init_N::Any
        Function, object, or number used to set the initial N(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_N(o, mdp, s, a)` will be called.
        If this is a number, N will be set to that number
        default: 0

    virtual_loss::Float64:
        A temporary loss added to the UCB score for nodes that are currently being
        evaluated by some threads. This can be used to encourage threads to explore
        broader parts of the search space. Relevant when running MCTS with multiple threads.
        default: 0.0

    reuse_tree::Bool:
        If this is true, the tree information is re-used for calculating the next plan.
        Of course, clear_tree! can always be called to override this.
        default: false

    enable_tree_vis::Bool:
        If this is true, extra information needed for tree visualization will
        be recorded. If it is false, the tree cannot be visualized.
        default: false
"""
mutable struct MCTSSolver <: AbstractMCTSSolver
    n_iterations::Int64
    max_time::Float64
    depth::Int64
    exploration_constant::Float64
    rng::AbstractRNG
    estimate_value::Any
    init_Q::Any
    init_N::Any
    virtual_loss::Float64
    reuse_tree::Bool
    enable_tree_vis::Bool
end

"""
    MCTSSolver()

Use keyword arguments to specify values for the fields.
"""
function MCTSSolver(;n_iterations::Int64=100,
                     max_time::Float64=Inf,
                     depth::Int64=10,
                     exploration_constant::Float64=1.0,
                     rng=Random.GLOBAL_RNG,
                     estimate_value=RolloutEstimator(RandomSolver(rng)),
                     init_Q=0.0,
                     init_N=0,
                     virtual_loss::Float64=0.0,
                     reuse_tree::Bool=false,
                     enable_tree_vis::Bool=false)
    return MCTSSolver(n_iterations, max_time, depth, exploration_constant, rng,
                      estimate_value, init_Q, init_N, virtual_loss, reuse_tree,
                      enable_tree_vis)
end


mutable struct MCTSTree{S,A}
    state_map::Dict{S,Int}

    # These vectors have one entry for each state node
    child_ids::Vector{Vector{Int}}
    total_n::Vector{Int}
    s_labels::Vector{S}

    # These vectors have one entry for each action node
    n::Vector{Int}
    q::Vector{Float64}
    a_labels::Vector{A}

    _vis_stats::Union{Nothing, Dict{Pair{Int,Int}, Int}} # Maps (said=>sid)=>number of transitions. THIS MAY CHANGE IN THE FUTURE

    # Locks and others needed for multi-threaded MCTS.
    # TODO(kykim): Also support the lock-free approach.
    state_locks::Vector{ReentrantLock}  # Tree parallelism with node lock.
    node_insertion_lock::ReentrantLock
    vis_stats_lock::ReentrantLock
    # Action nodes currently being evaluated. Used for applying virtual loss.
    a_selected::Set{Int}

    function MCTSTree{S,A}(sz::Int=1000) where {S,A}
        sz = min(sz, 100_000)

        return new(Dict{S, Int}(),

                   sizehint!(Vector{Int}[], sz),
                   sizehint!(Int[], sz),
                   sizehint!(S[], sz),

                   sizehint!(Int[], sz),
                   sizehint!(Float64[], sz),
                   sizehint!(A[], sz),
                   Dict{Pair{Int,Int},Int}(),

                   sizehint!(ReentrantLock[], sz),
                   ReentrantLock(),
                   ReentrantLock(),
                   Set{Int}(),
                  )
    end
end

Base.isempty(t::MCTSTree) = isempty(t.state_map)
state_nodes(t::MCTSTree) = (StateNode(t, id) for id in 1:length(t.total_n))

struct StateNode{S,A}
    tree::MCTSTree{S,A}
    id::Int
end
StateNode(tree::MCTSTree{S}, s::S) where S = StateNode(tree, tree.state_map[s])

"""
    get_state_node(tree::MCTSTree, s)

Return the StateNode in the tree corresponding to s.
"""
get_state_node(tree::MCTSTree, s) = StateNode(tree, s)


# Accessors for state nodes
@inline state(n::StateNode) = n.tree.s_labels[n.id]
@inline total_n(n::StateNode) = n.tree.total_n[n.id]
@inline child_ids(n::StateNode) = n.tree.child_ids[n.id]
@inline children(n::StateNode) = (ActionNode(n.tree, id) for id in child_ids(n))

struct ActionNode{S,A}
    tree::MCTSTree{S,A}
    id::Int
end

# Accessors for action nodes
@inline POMDPs.action(n::ActionNode) = n.tree.a_labels[n.id]
@inline n(n::ActionNode) = n.tree.n[n.id]
@inline q(n::ActionNode) = n.tree.q[n.id]


mutable struct MCTSPlanner{P<:Union{MDP,POMDP}, S, A, SE, RNG} <: AbstractMCTSPlanner{P}
    solver::MCTSSolver # contains the solver parameters
    mdp::P # model
    tree::Union{Nothing,MCTSTree{S,A}} # the search tree
    solved_estimate::SE
    rng::RNG
end


function MCTSPlanner(solver::MCTSSolver, mdp::Union{POMDP,MDP})
    tree = MCTSTree{statetype(mdp), actiontype(mdp)}(solver.n_iterations)
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return MCTSPlanner(solver, mdp, tree, se, solver.rng)
end


"""
Delete existing decision tree.
"""
function clear_tree!(p::MCTSPlanner{S,A}) where {S,A} p.tree = nothing end

"""
    get_state_node(tree::MCTSTree, s, planner::MCTSPlanner)

Return the StateNode in the tree corresponding to s. If there is no such node, add it using the planner.
"""
function get_state_node(tree::MCTSTree, s, planner::MCTSPlanner)
    if haskey(tree.state_map, s)
        return StateNode(tree, s)
    else
        # return lock(() -> insert_node!(tree, planner, s), tree.node_insertion_lock)
        return insert_node!(tree, planner, s)
    end
end


# No computation is done in solve - the solver is just given the mdp model that it will work with
POMDPs.solve(solver::MCTSSolver, mdp::Union{POMDP,MDP}) = MCTSPlanner(solver, mdp)

@POMDP_require POMDPs.action(policy::AbstractMCTSPlanner, state) begin
    @subreq simulate(policy, state, policy.solver.depth)
end

function POMDPModelTools.action_info(p::AbstractMCTSPlanner, s)
    println("Start action_info")
    tree = plan!(p, s)
    best = best_sanode_Q(StateNode(tree, s))
    return action(best), (tree=tree,)
end

POMDPs.action(p::AbstractMCTSPlanner, s) = first(action_info(p, s))

"""
Query the tree for a value estimate at state s. If the planner does not already have a tree, run the planner first.
"""
function POMDPs.value(planner::MCTSPlanner, s)
    if planner.tree == nothing
        plan!(planner, s)
    end
    return value(planner.tree, s)
end


function POMDPs.value(tr::MCTSTree, s)
    id = get(tr.state_map, s, 0)
    if id == 0
        error("State $s not present in MCTS tree.")
    end
    return maximum(q(san) for san in children(StateNode(tr, id)))
end


function POMDPs.value(planner::MCTSPlanner{<:Union{POMDP,MDP}, S, A}, s::S, a::A) where {S,A}
    if planner.tree == nothing
        plan!(planner, s)
    end
    return value(planner.tree, s, a)
end


function POMDPs.value(tr::MCTSTree{S,A}, s::S, a::A) where {S,A}
    for san in children(StateNode(tr, s)) # slow search through children
        if action(san) == a
            return q(san)
        end
    end
end


"""
Build tree and store it in the planner.
"""
function plan!(planner::AbstractMCTSPlanner, s)
    tree = build_tree(planner, s)
    planner.tree = tree
    return tree
end


function build_tree(planner::AbstractMCTSPlanner, s)
    n_iterations = planner.solver.n_iterations
    depth = planner.solver.depth

    if planner.solver.reuse_tree
        tree = planner.tree
    else
        tree = MCTSTree{statetype(planner.mdp), actiontype(planner.mdp)}(n_iterations)
    end

    sid = get(tree.state_map, s, 0)
    if sid == 0
        root = lock(() -> insert_node!(tree, planner, s), tree.node_insertion_lock)
        # root = insert_node!(tree, planner, s)
    else
        root = StateNode(tree, sid)
    end

    println("Before sim start")
    timeout_us = CPUtime_us() + planner.solver.max_time * 1e6
    sim_channel = Channel{Task}(min(1000, n_iterations), spawn=true) do channel
        for n in 1:n_iterations
            put!(channel, Threads.@spawn simulate(planner, root, depth, timeout_us, n))
            # println("Put ", n, ", ", length(channel.data))
        end
        println("Channel task finished ", channel)
    end

    counter = 0
    for sim_task in sim_channel
        counter += 1
        CPUtime_us() > timeout_us && break
        try
            println("Fetch ", counter)
            fetch(sim_task)  # Throws a TaskFailedException if failed.
        catch err
            throw(err.task.exception)  # Throw the underlying exception.
        end
    end

    return tree
end


function simulate(planner::AbstractMCTSPlanner, node::StateNode, depth::Int64, timeout_us::Float64=0.0, counter::Int64=0)
    println("simulate ", counter)
    mdp = planner.mdp
    rng = planner.rng
    s = state(node)
    tree = node.tree
    solver = planner.solver

    # Once depth is zero return.
    if isterminal(planner.mdp, s)
        return 0.0
    elseif depth == 0 || (timeout_us > 0.0 && CPUtime_us() > timeout_us)
        return estimate_value(planner.solved_estimate, planner.mdp, s, depth)
    end

    # Pick action using UCT.
    sanode = lock(() -> best_sanode_UCB(node, solver.exploration_constant, solver.virtual_loss), tree.state_locks[node.id])
    # sanode = best_sanode_UCB(node, solver.exploration_constant, solver.virtual_loss)
    said = sanode.id

    # Transition to a new state.
    sp, r = @gen(:sp, :r)(mdp, s, action(sanode), rng)

    spid = get(tree.state_map, sp, 0)
    if spid == 0
        # Using a dedicated lock for this method is safe (and more efficient
        # than a single lock for the entire tree) since this is the only method
        # that expands the vector attributes of the tree, and only the lengths
        # (or the no. of nodes) matter i.e., it is okay for other threads to
        # update e.g., the q, n values so long as the lengths stay the same.
        # spn = lock(() -> insert_node!(tree, planner, sp), tree.node_insertion_lock)
        spn = insert_node!(tree, planner, sp)
        spid = spn.id
        q = r + discount(mdp) * estimate_value(planner.solved_estimate, planner.mdp, sp, depth - 1)
    else
        # TODO: Does making this multi-threaded make sense? Maybe not.
        q = r + discount(mdp) * simulate(planner, StateNode(tree, spid) , depth - 1, timeout_us, counter)
    end
    if solver.enable_tree_vis
        lock(() -> record_visit!(tree, said, spid), tree.vis_stats_lock)
        # record_visit!(tree, said, spid)
    end

    function backpropagate(tree, sid, said, q)
        tree.total_n[sid] += 1
        tree.n[said] += 1
        tree.q[said] += (q - tree.q[said]) / tree.n[said] # moving average of Q value
        delete!(tree.a_selected, said)
    end
    # TODO(kykim): Consider using state-action lock.
    lock(() -> backpropagate(tree, node.id, said, q), tree.state_locks[node.id])
    # backpropagate(tree, node.id, said, q)

    return q
end


@POMDP_require simulate(planner::AbstractMCTSPlanner, s, depth::Int64, timeout_us::Float64, counter::Int64) begin
    mdp = planner.mdp
    P = typeof(mdp)
    S = statetype(P)
    A = actiontype(P)
    @req discount(::P)
    @req isterminal(::P, ::S)
    @subreq insert_node!(planner, s)
    @subreq estimate_value(planner.solved_estimate, mdp, s, depth)
    @req gen(::P, ::S, ::A, ::typeof(planner.rng)) # XXX this is not exactly right - it could be satisfied with transition
    @req isequal(::S, ::S) # for hasnode
    @req hash(::S) # for hasnode
end


function insert_node!(tree::MCTSTree, planner::MCTSPlanner, s)
    push!(tree.s_labels, s)
    tree.state_map[s] = length(tree.s_labels)
    push!(tree.child_ids, [])
    total_n = 0
    for a in actions(planner.mdp, s)
        n = init_N(planner.solver.init_N, planner.mdp, s, a)
        total_n += n
        push!(tree.n, n)
        push!(tree.q, init_Q(planner.solver.init_Q, planner.mdp, s, a))
        push!(tree.a_labels, a)
        push!(last(tree.child_ids), length(tree.n))
    end
    push!(tree.total_n, total_n)
    sid = length(tree.total_n)
    push!(tree.state_locks, ReentrantLock())
    return StateNode(tree, sid)
end


@POMDP_require insert_node!(tree::MCTSTree, planner::AbstractMCTSPlanner, s) begin
    # From the StateNode constructor
    P = typeof(planner.mdp)
    A = actiontype(P)
    S = typeof(s)
    IQ = typeof(planner.solver.init_Q)
    if !(IQ <: Number) && !(IQ <: Function)
        @req init_Q(::IQ, ::P, ::S, ::A)
    end
    IN = typeof(planner.solver.init_N)
    if !(IN <: Number) && !(IN <: Function)
        @req init_N(::IN, ::P, ::S, ::A)
    end
    @req actions(::P, ::S)
    as = actions(planner.mdp, s)
    @req isequal(::S, ::S) # for tree[s]
    @req hash(::S) # for tree[s]
end


function record_visit!(tree::MCTSTree, said::Int, spid::Int)
    vs = tree._vis_stats
    if !haskey(vs, said=>spid)
        vs[said=>spid] = 0
    end
    vs[said=>spid] += 1
end


"""
Return the best action based on the Q score
"""
function best_sanode_Q(snode::StateNode)
    best_Q = -Inf
    best = first(children(snode))
    for sanode in children(snode)
        if q(sanode) > best_Q
            best_Q = q(sanode)
            best = sanode
        end
    end
    return best
end


"""
Return the best action node based on the UCB score with exploration constant c
"""
function best_sanode_UCB(snode::StateNode, c::Float64, virtual_loss::Float64=0.0)
    tree = snode.tree
    best_UCB = -Inf
    best = first(children(snode))
    sn = total_n(snode)
    for sanode in children(snode)
        # if sn==0, log(sn) = -Inf. We want to avoid this.
        # in most cases, if n(sanode)==0, UCB will be Inf, which is desired,
        # but if sn==1 as well, then we have 0/0, which is NaN
        if c == 0 || sn == 0 || (sn == 1 && n(sanode) == 0)
            UCB = q(sanode)
        else
            vloss = (sanode.id in tree.a_selected ? virtual_loss : 0.0)
            UCB = q(sanode) + c*sqrt(log(sn)/n(sanode)) - vloss
        end

        if isnan(UCB)
            @show sn
            @show n(sanode)
            @show q(sanode)
        end

        @assert !isnan(UCB)
        @assert !isequal(UCB, -Inf)

        if UCB > best_UCB
            best_UCB = UCB
            best = sanode
        end
    end
    push!(tree.a_selected, best.id)
    return best
end
