/**
 * @file Revolve.cpp
 * @brief Implementation of the optimal binomial checkpointing scheduler
 */

#include "revolve/Revolve.hpp"
#include <cstdio>
#include <climits>

namespace SEM {

// =============================================================================
// Static utility functions
// =============================================================================

int Revolve::MaxRange(int snaps, int reps)
{
    if (snaps < 0 || reps < 0) { return -1; }

    double res = 1.0;
    for (int i = 1; i <= reps; i++)
    {
        res *= (snaps + i);
        res /= i;
        if (res > INT_MAX) { return INT_MAX; }
    }
    return static_cast<int>(res);
}

int Revolve::NumForw(int steps, int snaps)
{
    if (snaps < 1) { return -1; }

    int reps = 0;
    int range = 1;
    while (range < steps)
    {
        reps += 1;
        range = range * (reps + snaps) / reps;
    }
    if (reps > kMaxReps) { return -1; }

    int num = reps * steps - range * reps / (snaps + 1);
    return num;
}

double Revolve::Expense(int steps, int snaps)
{
    if (snaps < 1 || steps < 1) { return -1.0; }

    int num = NumForw(steps, snaps);
    if (num == -1) { return -1.0; }
    return static_cast<double>(num) / steps;
}

int Revolve::Adjust(int steps)
{
    int snaps = 1;
    int reps = 1;
    int s = 0;

    while (MaxRange(snaps + s, reps + s) > steps) { s--; }
    while (MaxRange(snaps + s, reps + s) < steps) { s++; }

    snaps += s;
    reps += s;
    s = -1;

    while (MaxRange(snaps, reps) >= steps)
    {
        if (snaps > reps)
        {
            snaps -= 1;
            s = 0;
        }
        else
        {
            reps -= 1;
            s = 1;
        }
    }

    if (s == 0) { snaps += 1; }
    if (s == 1) { reps += 1; }

    return snaps;
}

// =============================================================================
// Constructor
// =============================================================================

Revolve::Revolve(int steps, int snaps, bool verbose)
    : check_(-1)
    , capo_(0)
    , fine_(steps)
    , old_capo_(0)
    , snaps_(snaps)
    , verbose_(verbose)
    , turn_(0)
    , old_fine_(0)
    , old_snaps_(snaps)
    , advances_(0)
    , takeshots_(0)
    , commands_(0)
    , error_(RevolveError::None)
{
    for (int i = 0; i < kMaxCheckpoints; i++) { ch_[i] = 0; }
    ch_[0] = capo_ - 1;
}

// =============================================================================
// Main scheduling function
// =============================================================================

RevolveAction Revolve::Next()
{
    commands_ += 1;

    // Error checks
    if (check_ < -1 || capo_ > fine_)
    {
        error_ = RevolveError::None;
        return RevolveAction::Error;
    }

    // Initialize on first call
    if (check_ == -1 && capo_ < fine_)
    {
        turn_ = 0;
        ch_[0] = capo_ - 1;
    }

    old_capo_ = capo_;

    switch (fine_ - capo_)
    {
    case 0:
        // Reduce capo to previous checkpoint, unless done
        if (check_ == -1 || capo_ == ch_[0])
        {
            check_ -= 1;
            return RevolveAction::Terminate;
        }
        else
        {
            capo_ = ch_[check_];
            old_fine_ = fine_;
            return RevolveAction::Restore;
        }

    case 1:
        // Combined forward/adjoint step
        fine_ -= 1;
        if (check_ >= 0 && ch_[check_] == capo_)
        {
            check_ -= 1;
        }
        if (turn_ == 0)
        {
            turn_ = 1;
            old_fine_ = fine_;
            return RevolveAction::FirstTurn;
        }
        else
        {
            old_fine_ = fine_;
            return RevolveAction::YouTurn;
        }

    default:
        // Main scheduling logic
        if (check_ == -1 || ch_[check_] != capo_)
        {
            // Need to take a snapshot at current capo
            check_ += 1;

            if (check_ >= kMaxCheckpoints)
            {
                error_ = RevolveError::CheckExceedsLimit;
                return RevolveAction::Error;
            }
            if (check_ + 1 > snaps_)
            {
                error_ = RevolveError::CheckExceedsSnaps;
                return RevolveAction::Error;
            }

            ch_[check_] = capo_;

            if (check_ == 0)
            {
                // First snapshot: initialize counters
                advances_ = 0;
                takeshots_ = 0;
                commands_ = 1;
                old_snaps_ = snaps_;

                if (snaps_ > kMaxCheckpoints)
                {
                    error_ = RevolveError::SnapsExceedsLimit;
                    return RevolveAction::Error;
                }
                if (verbose_)
                {
                    int num = NumForw(fine_ - capo_, snaps_);
                    if (num == -1)
                    {
                        error_ = RevolveError::NumForwError;
                        return RevolveAction::Error;
                    }
                    std::printf("  [Revolve] predicted forward steps: %d\n", num);
                    std::printf("  [Revolve] slowdown factor: %.4f\n\n",
                                static_cast<double>(num) / (fine_ - capo_));
                }
            }

            takeshots_ += 1;
            old_fine_ = fine_;
            return RevolveAction::TakeShot;
        }
        else
        {
            // Checkpoint at capo already exists: compute advance distance
            if (old_fine_ < fine_ && snaps_ == check_ + 1)
            {
                error_ = RevolveError::FineEnhanced;
                return RevolveAction::Error;
            }

            int ds = snaps_ - check_;
            if (ds < 1)
            {
                error_ = RevolveError::CheckExceedsSnaps;
                return RevolveAction::Error;
            }

            // Find minimum reps such that range >= fine - capo
            int reps = 0;
            int range = 1;
            while (range < fine_ - capo_)
            {
                reps += 1;
                range = range * (reps + ds) / reps;
            }
            if (reps > kMaxReps)
            {
                error_ = RevolveError::RepsExceedsLimit;
                return RevolveAction::Error;
            }

            if (snaps_ != old_snaps_)
            {
                if (snaps_ > kMaxCheckpoints)
                {
                    error_ = RevolveError::SnapsExceedsLimit;
                    return RevolveAction::Error;
                }
            }

            // Compute binomial coefficients for optimal advance
            // range = beta(ds, reps) >= fine - capo
            // bino1 = beta(ds, reps-1)
            int bino1 = range * reps / (ds + reps);

            // bino2 = beta(ds-1, reps-1)
            int bino2 = (ds > 1) ? bino1 * ds / (ds + reps - 1) : 1;

            // bino3 = beta(ds-2, reps-1)
            int bino3;
            if (ds == 1)
                bino3 = 0;
            else
                bino3 = (ds > 2) ? bino2 * (ds - 1) / (ds + reps - 2) : 1;

            // bino4 = beta(ds, reps-2)
            int bino4 = bino2 * (reps - 1) / ds;

            // bino5 = beta(ds-3, reps)
            int bino5;
            if (ds < 3)
                bino5 = 0;
            else
                bino5 = (ds > 3) ? bino3 * (ds - 2) / reps : 1;

            // bino6 = beta(ds-1, reps)
            int bino6 = bino1 * ds / reps;

            // Determine optimal advance using Kowarz's refinement
            int old_capo_val = capo_;
            if (fine_ - capo_ <= bino1 + bino3)
                capo_ += bino4;
            else if (fine_ - capo_ < bino1 + bino2)
                capo_ = fine_ - bino2 - bino3;
            else if (fine_ - capo_ <= bino1 + bino2 + bino5)
                capo_ += bino1 - bino3;
            else
                capo_ = fine_ - bino6;

            // Ensure at least one step of progress
            if (capo_ == old_capo_val)
            {
                capo_ = old_capo_val + 1;
            }

            advances_ += capo_ - old_capo_val;
            old_fine_ = fine_;
            old_capo_ = old_capo_val;
            return RevolveAction::Advance;
        }
    }
}

}  // namespace SEM
