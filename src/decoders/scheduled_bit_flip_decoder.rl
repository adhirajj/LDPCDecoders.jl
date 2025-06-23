using SparseArrays
using LinearAlgebra

# STATE FOR STORING BIT-FLIP OTS COMPUTATIONS
mutable struct BitFlipOTSState
    decisions::Vector{Int}          # Current error estimate x̂
    syndrome_estimate::Vector{Bool} # Current syndrome estimate σ̂
    syndrome_mismatch::Vector{Bool} # β = σ̂ ⊕ σ
    unsatisfied_counts::Vector{Int} # α = HT_Z β (count of unsatisfied checks per variable)
    oscillations::Vector{Int}       # Oscillation counter σ (tracks bit flips)
    prior_decisions::Vector{Int}    # Previous decisions for oscillation tracking
    biased_nodes::Set{Int}         # Currently biased nodes
    bias_values::Vector{Float64}   # Bias values for each node
    
    # Pre-allocated buffers
    int_syndrome_buffer::Vector{Int} # Integer buffer for matrix multiplication
    best_decisions::Vector{Int}      # Best solution found so far
end

# MAIN BIT-FLIP OTS DECODER STRUCT
struct BitFlipOTSDecoder <: AbstractDecoder
    max_iters::Int      # Maximum number of iterations
    s::Int             # Number of stabilizers (rows in H)
    n::Int             # Number of qubits (columns in H)
    n1n2::Int          # Number of VV-type variable nodes (for scheduled updates)
    T::Int             # Biasing period (how often to apply bias)
    bias_strength::Float64  # Strength of bias to apply
    sparse_H::SparseMatrixCSC{Bool,Int}  # Sparse parity check matrix HZ
    sparse_HT::SparseMatrixCSC{Bool,Int} # Transposed parity check matrix
    scratch::BitFlipOTSState # Working state for computations
    
    var_neighbors::Vector{Vector{Int}}  # Pre-computed variable neighbors |Nvi|
end

# BIT-FLIP OTS STATE INITIALIZATION
function initialize_bitflip_ots_state(H::SparseMatrixCSC, n::Int)
    return BitFlipOTSState(
        zeros(Int, n),           # decisions
        zeros(Bool, size(H,1)),  # syndrome_estimate  
        zeros(Bool, size(H,1)),  # syndrome_mismatch
        zeros(Int, n),           # unsatisfied_counts
        zeros(Int, n),           # oscillations
        zeros(Int, n),           # prior_decisions
        Set{Int}(),              # biased_nodes
        zeros(Float64, n),       # bias_values
        zeros(Int, size(H,1)),   # int_syndrome_buffer
        zeros(Int, n)            # best_decisions
    )
end

# CONSTRUCTOR FOR BIT-FLIP OTS DECODER
function BitFlipOTSDecoder(H::Union{SparseMatrixCSC{Bool,Int}, BitMatrix}, n1::Int, n2::Int, max_iters::Int=200; T::Int=9, bias_strength::Float64=0.6)
    s, n = size(H)
    sparse_H = sparse(H)
    sparse_HT = sparse(H')
    n1n2 = n1 * n2  # Number of VV-type nodes as per paper
    scratch = initialize_bitflip_ots_state(sparse_H, n)
    
    # Pre-compute neighbors for each variable node to get |Nvi|
    var_neighbors = [Vector{Int}() for _ in 1:n]
    
    for j in 1:n
        for idx in nzrange(sparse_H, j)
            i = rowvals(sparse_H)[idx]
            if nonzeros(sparse_H)[idx]
                push!(var_neighbors[j], i)
            end
        end
    end
    
    return BitFlipOTSDecoder(max_iters, s, n, n1n2, T, bias_strength, sparse_H, sparse_HT, scratch, var_neighbors)
end

# RESET STATE BETWEEN DECODINGS
function reset!(decoder::BitFlipOTSDecoder)
    state = decoder.scratch
    fill!(state.decisions, 0)  # x̂ ← 0
    fill!(state.syndrome_estimate, false)
    fill!(state.syndrome_mismatch, false)
    fill!(state.unsatisfied_counts, 0)
    fill!(state.oscillations, 0)
    fill!(state.prior_decisions, 0)
    fill!(state.bias_values, 0)
    empty!(state.biased_nodes)
    fill!(state.int_syndrome_buffer, 0)
    fill!(state.best_decisions, 0)
    return decoder
end

# COMPUTE SYNDROME ESTIMATE: σ̂ ← HZx̂(mod 2)
function compute_syndrome_estimate!(decoder::BitFlipOTSDecoder, state::BitFlipOTSState)
    fill!(state.int_syndrome_buffer, 0)
    mul!(state.int_syndrome_buffer, decoder.sparse_H, state.decisions)
    
    for i in 1:decoder.s
        state.syndrome_estimate[i] = mod(state.int_syndrome_buffer[i], 2) == 1
    end
end

# COMPUTE SYNDROME MISMATCH: β ← σ̂ ⊕ σ
function compute_syndrome_mismatch!(state::BitFlipOTSState, target_syndrome::Vector{Bool})
    for i in 1:length(target_syndrome)
        state.syndrome_mismatch[i] = state.syndrome_estimate[i] ⊻ target_syndrome[i]
    end
end

# COMPUTE UNSATISFIED COUNTS: α = HT_Z β
function compute_unsatisfied_counts!(decoder::BitFlipOTSDecoder, state::BitFlipOTSState)
    fill!(state.unsatisfied_counts, 0)
    mul!(state.unsatisfied_counts, decoder.sparse_HT, state.syndrome_mismatch)
end

# CHECK IF CONVERGED: HZx̂ = σ (mod 2)
function check_converged(decoder::BitFlipOTSDecoder, state::BitFlipOTSState, syndrome::Vector{Bool})
    compute_syndrome_estimate!(decoder, state)
    for i in 1:decoder.s
        if state.syndrome_estimate[i] != syndrome[i]
            return false
        end
    end
    return true
end

# APPLY BIAS TO FLIPPING THRESHOLD
function get_biased_threshold(decoder::BitFlipOTSDecoder, state::BitFlipOTSState, j::Int)
    degree = length(decoder.var_neighbors[j])
    base_threshold = div(degree, 2)
    
    if j in state.biased_nodes
        # Apply bias to make flipping more likely for oscillating nodes
        bias_adjustment = state.bias_values[j] * degree
        return base_threshold - bias_adjustment
    else
        return base_threshold
    end
end

# UPDATE OSCILLATIONS AND APPLY BIASING STRATEGY
function update_oscillations_and_bias!(decoder::BitFlipOTSDecoder, state::BitFlipOTSState, iter::Int)
    # Update oscillation counters
    if iter > 1
        for j in 1:decoder.n
            if state.decisions[j] != state.prior_decisions[j]
                state.oscillations[j] += 1
            end
        end
    end
    state.prior_decisions .= state.decisions
    
    # Apply biasing every T iterations
    if iter % decoder.T == 0 && maximum(state.oscillations) > 0
        # Reset previous biases
        empty!(state.biased_nodes)
        fill!(state.bias_values, 0)
        
        # Find nodes with maximum oscillations
        max_osc = maximum(state.oscillations)
        if max_osc > 0
            oscillating_nodes = findall(x -> x == max_osc, state.oscillations)
            
            # Bias the most oscillating nodes
            for j in oscillating_nodes
                push!(state.biased_nodes, j)
                state.bias_values[j] = decoder.bias_strength
                # Reset oscillation counter for biased node
                state.oscillations[j] = 0
            end
            
            # Also bias nodes with high unsatisfied counts that aren't oscillating much
            if length(oscillating_nodes) < 3  # Don't bias too many nodes at once
                for j in 1:decoder.n
                    if j ∉ state.biased_nodes && state.oscillations[j] > 0
                        degree = length(decoder.var_neighbors[j])
                        if state.unsatisfied_counts[j] > div(degree, 2) + 1
                            push!(state.biased_nodes, j)
                            state.bias_values[j] = decoder.bias_strength * 0.5  # Weaker bias
                            break  # Only add one more
                        end
                    end
                end
            end
        end
    end
end

# MAIN DECODING ALGORITHM - Bit-flipping with OTS bias
function decode!(decoder::BitFlipOTSDecoder, syndrome::Vector{Bool})
    state = decoder.scratch
    reset!(decoder)
    
    # Initialize tracking for best solution
    fill!(state.best_decisions, 0)
    best_mismatch = decoder.s
    best_weight = decoder.n
    
    for iter in 1:decoder.max_iters
        # Check convergence first
        if check_converged(decoder, state, syndrome)
            state.best_decisions .= state.decisions
            return state.best_decisions, true
        end
        
        # Track best solution
        compute_syndrome_estimate!(decoder, state)
        compute_syndrome_mismatch!(state, syndrome)
        mismatch = sum(state.syndrome_mismatch)
        weight = sum(state.decisions)
        
        if mismatch < best_mismatch || (mismatch == best_mismatch && weight < best_weight)
            best_mismatch = mismatch
            best_weight = weight
            state.best_decisions .= state.decisions
        end
        
        # Update oscillations and apply biasing strategy
        update_oscillations_and_bias!(decoder, state, iter)
        
        # Compute current state for this iteration
        compute_syndrome_estimate!(decoder, state)
        compute_syndrome_mismatch!(state, syndrome)
        compute_unsatisfied_counts!(decoder, state)
        
        # PHASE 1: Update VV-type variable nodes (i = 1 to n1n2) with bias
        for i in 1:decoder.n1n2
            threshold = get_biased_threshold(decoder, state, i)
            if state.unsatisfied_counts[i] > threshold
                state.decisions[i] = 1 - state.decisions[i]  # Flip bit
            end
        end
        
        # Recompute after VV updates
        compute_syndrome_estimate!(decoder, state)
        compute_syndrome_mismatch!(state, syndrome)
        compute_unsatisfied_counts!(decoder, state)
        
        # PHASE 2: Update CC-type variable nodes (i = n1n2 + 1 to n) with bias
        if !check_converged(decoder, state, syndrome)
            for i in (decoder.n1n2 + 1):decoder.n
                threshold = get_biased_threshold(decoder, state, i)
                if state.unsatisfied_counts[i] > threshold
                    state.decisions[i] = 1 - state.decisions[i]  # Flip bit
                end
            end
        end
    end
    
    # Return best solution found
    return state.best_decisions, false
end

# Export the main types and functions
export BitFlipOTSDecoder, decode!, reset!