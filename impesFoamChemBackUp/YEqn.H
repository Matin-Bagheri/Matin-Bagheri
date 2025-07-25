#include "phaseUtil.H"

forAll(mesh.C(), celli)
{

    // 1. Gather compositions
    List<scalar> X(speciesNames.size());
    List<scalar> Y(speciesNames.size());
    forAll(speciesNames, i)
    {
        X[i] = Xfields[i][celli];
        Y[i] = Yfields[i][celli];
    }
	
	scalar sb = Sb[celli];
	if (sb < 0.999 && sb > 0.001)
	{
    // 2. Calculate phase molar densities
    scalar a_molarDensity = calcMolarDensity(X);
    scalar b_molarDensity = calcMolarDensity(Y);

    // 3. Calculate global mole fractions Z
    List<scalar> Z(speciesNames.size());
	scalar b = sb * b_molarDensity / (sb*b_molarDensity + (1-sb)*a_molarDensity);
    
    forAll(speciesNames, i)
    {
        Z[i] = X[i] * (1.0 - b)  + Y[i] * b;
    }

    // 4. Solve Rachford-Rice equation for beta using Halley's method
    scalar beta = 0.5; // initial guess
    scalar tol = 1e-8;
	scalar relaxationFactor = 0.25;
    int maxIter = 50;
	scalar betaUpper = 1.0;
	scalar betaLower = 0.0;
    for (int iter=0; iter<maxIter; iter++)
    {
        scalar f = 0.0;
		scalar df = 0.0;
		
		forAll(speciesNames, i)
		{
			scalar denom = 1.0 + beta * (K_values[i]-1.0);
			f += Z[i] * (K_values[i]-1.0) / denom;
			df -= Z[i] * std::pow(K_values[i]-1.0, 2.0) / std::pow(denom, 2.0);
		}
		
		scalar betaNew = beta - relaxationFactor * f/df;
		if (betaNew < betaLower || betaNew > betaUpper) 
		{
			betaNew = 0.5*(betaLower + betaUpper);
		}
		
		f = 0.0;
		df = 0.0;
		
		forAll(speciesNames, i)
		{
			scalar denom = 1.0 + beta * (K_values[i]-1.0);
			f += Z[i] * (K_values[i]-1.0) / denom;
			df -= Z[i] * std::pow(K_values[i]-1.0, 2.0) / std::pow(denom, 2.0);
		}
		    // Update bounds
		if (f > 0) betaLower = betaNew;
		else betaUpper = betaNew;
		if (std::abs(betaNew - beta) < tol) break;
		beta = betaNew;
    }

    // 5. Update Y and X using new beta
    forAll(speciesNames, i)
    {
        X[i] = Z[i] / (1 + beta * (K_values[i]-1));
        Y[i] = X[i] * K_values[i];
    }

    // 6. Recalculate phase molar densities with updated X and Y
    a_molarDensity = calcMolarDensity(X);
    b_molarDensity = calcMolarDensity(Y);

    // 7. Update Sb with new densities
    sb = (beta / b_molarDensity) / (beta / b_molarDensity + (1.0 - beta) / a_molarDensity);

    // 8. Write updated values back to fields
    Sb[celli] = sb;
    forAll(speciesNames, i)
    {
        Xfields[i][celli] = X[i];
        Yfields[i][celli] = Y[i];
    }
}
}
// Correct boundary conditions after the loop
Sb.correctBoundaryConditions();
forAll(speciesNames, i)
{
    Xfields[i].correctBoundaryConditions();
    Yfields[i].correctBoundaryConditions();
}
