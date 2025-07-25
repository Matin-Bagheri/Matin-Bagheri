#include "phaseUtil.H"
#include "RachfordRice.H"


forAll(mesh.C(), celli)
{

    // 1. Gather compositions
    List<scalar> X(speciesNames.size());
    List<scalar> Y(speciesNames.size());
    List<scalar> G(speciesNames.size());
	
    forAll(speciesNames, i)
    {
        X[i] = Xfields[i][celli];
        Y[i] = Yfields[i][celli];
        G[i] = Gfields[i][celli];
    }
	
	scalar sb = Sb[celli];

	// 2. Calculate phase molar densities
	scalar a_molDensity = calcMolarDensity(X);
	scalar b_molDensity = calcMolarDensity(Y);

	// 3. Calculate global mole fractions Z
	List<scalar> Z(speciesNames.size());
	//scalar b = sb * b_molarDensity / (sb*b_molarDensity + (1-sb)*a_molarDensity);
	scalar G_sum = 0.0;
	forAll(speciesNames, i)
	{
		//Z[i] = X[i] * (1.0 - b)  + Y[i] * b;
		G_sum += G[i];
	}
	
	forAll(speciesNames, i)
	{
		Z[i] = G[i] / G_sum;
	}

	// 4. Solve Rachford-Rice equation for beta using Halley's method
	scalar beta = 0.5; // initial guess
	bool converged = rachfordRiceSolve(K_values, Z, beta, 1e-8, 1e-8, 50, 0.75);
	
	
	if (!converged) {
		// Optionally, you can skip this cell, or fall back to beta = sb, or clamp beta to [0,1]
		beta = min(max(beta, 0.0), 1.0);
	}
	
	

	// 5. Update Y and X using new beta
	forAll(speciesNames, i)
	{
		if (beta<0.0)
		{
			beta = 0.0;
			X[i] = Z[i];
			Y[i] = 0.0;
		}
		else if (beta>1.0)
		{
			beta = 1.0;
			X[i] = 0.0;
			Y[i] = Z[i];
		}
		else
		{
			X[i] = Z[i] / (1 + beta * (K_values[i]-1));
			Y[i] = X[i] * K_values[i];
		}
	}

	// 6. Recalculate phase molar densities with updated X and Y
	a_molDensity = calcMolarDensity(X);
	b_molDensity = calcMolarDensity(Y);
	

	// 7. Update Sb with new densities
	sb = (beta / b_molDensity) / (beta / b_molDensity + (1.0 - beta) / a_molDensity);

	// 8. Write updated values back to fields
	Sb[celli] = sb;
	forAll(speciesNames, i)
	{
		Xfields[i][celli] = X[i];
		Yfields[i][celli] = Y[i];
		Gfields[i][celli] = G[i];
	}

}
// Correct boundary conditions after the loop
Sb.correctBoundaryConditions();
forAll(speciesNames, i)
{
    Xfields[i].correctBoundaryConditions();
    Yfields[i].correctBoundaryConditions();
}
std::cout<<"min(Sb) = " << gMin(Sb)
         << ", max(Sb) = " << gMax(Sb) << std::endl;
		 
forAll(speciesNames, i)
{
	Info << "Component " << speciesNames[i]
         << ": min(X) = " << gMin(Xfields[i])
         << ", max(X) = " << gMax(Xfields[i]) << endl;
		 
	Info << "Component " << speciesNames[i]
         << ": min(Y) = " << gMin(Yfields[i])
         << ", max(Y) = " << gMax(Yfields[i]) << endl;
}