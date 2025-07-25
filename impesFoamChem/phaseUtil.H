#ifndef PHASEUTIL_H
#define PHASEUTIL_H

#include "scalar.H"
#include "List.H"

extern Foam::List<Foam::scalar> ZCI_global;
extern Foam::scalar T_global;
extern Foam::scalar p_global;
extern Foam::scalar R_global;

// Utility: Calculate phase molar density from composition using global T, p, R, ZCI
inline Foam::scalar calcMolarDensity(const Foam::List<Foam::scalar>& composition)
{
    Foam::List<Foam::scalar> molarVolume(ZCI_global.size());
    forAll(ZCI_global, i)
    {
        molarVolume[i] = ZCI_global[i] * R_global * T_global / p_global;
    }

    Foam::scalar denom = 0.0;
    forAll(composition, i)
    {
        denom += composition[i] * molarVolume[i];
    }
    denom = Foam::max(denom, Foam::SMALL); // Prevent division by zero
    return 1.0 / denom;
}

inline bool isPhaseSplitNeeded(const Foam::List<Foam::scalar>& Z, const Foam::List<Foam::scalar>& K_values)
{
    Foam::scalar S_L = 0.0;
    Foam::scalar S_V = 0.0;

    forAll(Z, i)
    {
        if (K_values[i] > Foam::SMALL)
        {
            Foam::scalar t1 = (K_values[i] - 1.0) / K_values[i] + 1.0;
            S_L += Z[i] * std::log(std::max(t1, Foam::SMALL));
            S_V += Z[i] * std::log(std::max(K_values[i], Foam::SMALL));
        }
    }

    return (S_L > 0.0 && S_V > 0.0); // True if two-phase, else false
}


#endif
