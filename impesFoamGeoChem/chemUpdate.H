{
// assuming you already have
// volScalarField Sb( ... );
// twoPhaseGeochemicalModel porousMedia( mesh, transportProperties, Sb, *phasea, *phaseb );

// 1) Run your porousMedia chemistry and grab the new field:
tmp<volScalarField> newSb = porousMedia.solutionSaturation();

// 2) Assign it back into your solver’s Sb:
Sb = 1-newSb;                    // calls volScalarField::operator=()
// 3) Enforce boundary conditions on the updated field:
Sb.correctBoundaryConditions();

}
