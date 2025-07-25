/*extern Foam::scalar p_global;
extern Foam::scalar T_global;
extern Foam::scalar R_global;
extern Foam::List<Foam::scalar> ZCI_global;*/
#include "phaseUtil.H"

forAll(mesh.C(), celli)
{
    // Prepare lists of composition for this cell
    Foam::List<Foam::scalar> Xa(speciesNames.size());
    Foam::List<Foam::scalar> Yb(speciesNames.size());
    forAll(speciesNames, i)
    {
        Xa[i] = Xfields[i][celli];
        Yb[i] = Yfields[i][celli];
    }

    // Compute densities using your utility function
    a_molarDensity[celli] = calcMolarDensity(Xa);
    b_molarDensity[celli] = calcMolarDensity(Yb);
}

forAll(speciesNames, i)
{
    // Phase composition fields (cell-centered)
    volScalarField& X = Xfields[i];
    volScalarField& Y = Yfields[i];

    // Phase molar densities (assume already calculated per cell)
    // If not, calculate or interpolate them
	scalar denom = p_global / (ZCI_global[i] * R_global * T_global);
    // G[i]: conserved variable, total moles of i per cell
	Gfields[i]= eps * (Sb * b_molarDensity * Y + (scalar(1.0) - Sb) * a_molarDensity * X) / denom;
					 
	forAll(mesh.boundary(), patchi)
	{
		if (isA< fixedValueFvPatchField<vector> >(Ua.boundaryField()[patchi]))
		{
			phia.boundaryFieldRef()[patchi]
				= Ua.boundaryField()[patchi] & mesh.Sf().boundaryField()[patchi];
		}
		if (isA< fixedValueFvPatchField<vector> >(Ub.boundaryField()[patchi]))
		{
			phib.boundaryFieldRef()[patchi]
				= Ub.boundaryField()[patchi] & mesh.Sf().boundaryField()[patchi];
		}
	}

    // F[i]: total face flux of i, using phase fluxes
    Ffields[i] = (fvc::interpolate(b_molarDensity * Y) * phib
               + fvc::interpolate(a_molarDensity * X) * phia) / denom;

    // The denominators normalize for phase disappearance (optional but recommended for robustness).
    // fvc::interpolate(Sb) gives face Sb for phib, etc.

    // Build and solve the finite volume equation
}

forAll(speciesNames, i)
{	
	Info << "solving GEqn" << endl;
	fvScalarMatrix GEqn
    (
        fvm::ddt(Gfields[i])
      + fvc::div(Ffields[i])
      ==
        - sourceTerm // for this species
    );

	
    GEqn.solve();
	//Gfields[i].max(0.0);

    Info << "Component " << speciesNames[i]
         << ": min(G) = " << gMin(Gfields[i])
         << ", max(G) = " << gMax(Gfields[i]) << endl;
		 
	Info << "Component " << speciesNames[i]
         << ": min(F) = " << gMin(Ffields[i])
         << ", max(F) = " << gMax(Ffields[i]) << endl;
}

Info << "Gas molar Density "<< ": min(Dg) = " 
	 << gMin(b_molarDensity)
	 << ", max(Dg) = " << gMax(b_molarDensity) << endl;
	 
Info << "water molar Density "<< ": min(Dw) = " 
	 << gMin(a_molarDensity)
	 << ", max(Dw) = " << gMax(a_molarDensity) << endl;