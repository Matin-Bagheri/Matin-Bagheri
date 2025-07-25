#ifndef RACHFORDRICE_H
#define RACHFORDRICE_H

#include "List.H"
#include "scalar.H"

bool rachfordRiceSolve(
    const Foam::List<Foam::scalar>& K_values,
    const Foam::List<Foam::scalar>& Z,
    Foam::scalar& beta,
    Foam::scalar tol = 1e-12,
    Foam::scalar tolF = 1e-12,
    int maxIter = 50,
    Foam::scalar relaxation = 1.0);

#endif
