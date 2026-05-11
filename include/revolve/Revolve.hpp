/**
 * @file Revolve.hpp
 * @brief Optimal binomial checkpointing scheduler for adjoint computations
 *
 * Self-contained implementation of the Revolve algorithm for optimal
 * checkpoint placement in adjoint (reverse-mode) time-stepping computations.
 *
 * The algorithm minimizes the number of forward recomputations given a fixed
 * number of checkpoint slots, using binomial coefficients to determine
 * optimal placement.
 *
 * Reference:
 *   Griewank, A. & Walther, A. (2000),
 *   "Algorithm 799: Revolve: An Implementation of Checkpointing for the
 *    Reverse or Adjoint Mode of Computational Differentiation",
 *   ACM Transactions on Mathematical Software, 26(1), 19-45.
 */

#ifndef SEM_REVOLVE_HPP
#define SEM_REVOLVE_HPP

#include <cstdint>

namespace SEM {

/// Actions returned by the Revolve scheduler
enum class RevolveAction
{
    Advance,    ///< Advance forward from capo to the new capo value
    TakeShot,   ///< Save checkpoint at slot indicated by check
    Restore,    ///< Restore checkpoint from slot indicated by check
    FirstTurn,  ///< First combined forward/adjoint step (at the end of forward sweep)
    YouTurn,    ///< Perform one adjoint step
    Terminate,  ///< Computation complete
    Error       ///< An error occurred (see info for details)
};

/// Error codes stored in info when RevolveAction::Error is returned
enum class RevolveError
{
    None = 0,
    CheckExceedsLimit = 10,   ///< checkpoint index exceeds internal limit
    CheckExceedsSnaps = 11,   ///< checkpoint index exceeds snaps
    NumForwError = 12,        ///< error in NumForw computation
    FineEnhanced = 13,        ///< fine was enhanced but all snaps used
    SnapsExceedsLimit = 14,   ///< snaps exceeds internal limit
    RepsExceedsLimit = 15     ///< repetition count exceeds internal limit
};

/**
 * @brief Optimal binomial checkpointing scheduler
 *
 * Usage:
 *   1. Create a Revolve object with total steps and number of checkpoint slots
 *   2. Call Revolve() repeatedly in a loop
 *   3. Respond to the returned action (Advance, TakeShot, Restore, YouTurn, etc.)
 *   4. Stop when Terminate is returned
 *
 * Example:
 *   Revolve rev(total_steps, num_checkpoints);
 *   while (true) {
 *       auto action = rev.Next();
 *       switch (action) {
 *       case RevolveAction::Advance:
 *           // run forward from rev.OldCapo() to rev.Capo()
 *           break;
 *       case RevolveAction::TakeShot:
 *           // save checkpoint to slot rev.Check()
 *           break;
 *       case RevolveAction::Restore:
 *           // restore checkpoint from slot rev.Check()
 *           break;
 *       case RevolveAction::FirstTurn:
 *       case RevolveAction::YouTurn:
 *           // perform one adjoint step at rev.Fine()
 *           break;
 *       case RevolveAction::Terminate:
 *           return;
 *       }
 *   }
 */
class Revolve
{
public:
    /// Maximum number of checkpoint slots supported
    static constexpr int kMaxCheckpoints = 256;

    /// Maximum number of repetitions supported
    static constexpr int kMaxReps = 256;

    /**
     * @brief Construct a Revolve scheduler
     * @param steps Total number of forward time steps
     * @param snaps Maximum number of checkpoint slots available
     * @param verbose If true, print prediction info on first call
     */
    Revolve(int steps, int snaps, bool verbose = false);

    /**
     * @brief Get the next action to perform
     * @return The action the caller should take
     */
    RevolveAction Next();

    /// Current checkpoint slot index (for TakeShot/Restore)
    int Check() const { return check_; }

    /// Current beginning of the subrange being processed
    int Capo() const { return capo_; }

    /// Previous capo value before the last Advance action
    int OldCapo() const { return old_capo_; }

    /// Current end of the subrange being processed
    int Fine() const { return fine_; }

    /// Error code (valid when Error action is returned)
    RevolveError GetError() const { return error_; }

    // === Static utility functions ===

    /**
     * @brief Compute the number of forward steps needed
     * @param steps Total number of time steps
     * @param snaps Number of checkpoint slots
     * @return Number of forward recomputations, or -1 on error
     */
    static int NumForw(int steps, int snaps);

    /**
     * @brief Compute the slowdown factor
     * @param steps Total number of time steps
     * @param snaps Number of checkpoint slots
     * @return Ratio of forward recomputations to total steps, or -1.0 on error
     */
    static double Expense(int steps, int snaps);

    /**
     * @brief Compute the maximum range coverable with given snaps and reps
     *
     * Returns C(snaps + reps, reps) = (snaps+reps)! / (snaps! * reps!)
     *
     * @param snaps Number of checkpoint slots
     * @param reps Number of repetitions
     * @return The binomial coefficient, or -1 on error
     */
    static int MaxRange(int snaps, int reps);

    /**
     * @brief Suggest a number of checkpoints for given step count
     *
     * Returns a value of snaps such that the increase in spatial complexity
     * approximately equals the increase in temporal complexity.
     *
     * @param steps Total number of time steps
     * @return Suggested number of checkpoint slots
     */
    static int Adjust(int steps);

private:
    int check_;       ///< Current checkpoint index
    int capo_;        ///< Start of current subrange
    int fine_;        ///< End of current subrange
    int old_capo_;    ///< Previous capo (before last advance)
    int snaps_;       ///< Maximum number of checkpoints
    bool verbose_;    ///< Whether to print info

    // Internal state
    int turn_;        ///< Turn counter (0 = first turn not yet taken)
    int old_fine_;    ///< Previous fine value
    int old_snaps_;   ///< Snaps value at initialization
    int advances_;    ///< Count of advance actions
    int takeshots_;   ///< Count of takeshot actions
    int commands_;    ///< Count of total commands
    int ch_[kMaxCheckpoints]; ///< Checkpoint stack: ch_[j] = step stored at slot j

    RevolveError error_; ///< Last error code
};

}  // namespace SEM

#endif  // SEM_REVOLVE_HPP
