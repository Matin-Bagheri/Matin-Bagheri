///////////////////////////////////////////////////////////////
////////////////////// TIMESTEP CONTROL ///////////////////////
///////////////////////////////////////////////////////////////

word CFL =  runTime.controlDict().lookupOrDefault<word>("CFL", "Coats");
if (adjustTimeStep) adjustTimeStep=true; // to remove warnings at compilation
int CFLint = -1;
if (CFL == "Coats") CFLint = 0;
else if (CFL == "Todd") CFLint = 1;
else if (CFL == "Courant") CFLint = 2;
else
{
    FatalErrorIn
        (
            "in createFields.H"
        )
        << "CFL condition unrecongnized : Coats, Todd and Courant available"
            << exit(FatalError);
}

/////////////////////////////////////////////////////////////////////////////
////////////////////////// VELOCITY - FLUXES ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/*
Info << nl << "Reading field U" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("U",dimensionSet(0,1,-1,0,0),vector::zero)
);

#include "createPhi.H"
surfaceScalarField phiP = phi;
*/

//////////////////////////////////////////////////////////////////
////////////////////// PRESSURE SATURATION ///////////////////////
//////////////////////////////////////////////////////////////////

Info << "Reading pressure field p" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info << nl << "Reading saturation field Sb" << endl;
volScalarField Sb
(
    IOobject
    (
        "Sb",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

scalar dSmax(runTime.controlDict().lookupOrDefault<scalar>("dSmax",0.));

//////////////////////////////////////////////////////////////////
////////////////////// TRANSPORT PROPERTIES //////////////////////
//////////////////////////////////////////////////////////////////

Info << nl << "Reading transportProperties" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

//- list that receives event files of event-based boundary conditions
//List<patchEventFile*> patchEventList;
//eventInfiltration::setEventFileRegistry(&patchEventList, Sb.name());

/////////////////////////////////////////////////////////////////////////////
/////////////////////////// PHASE MODEL CREATION ////////////////////////////
/////////////////////////////////////////////////////////////////////////////
//Info<< "Reading transportProperties\n" << endl;
//immiscibleIncompressibleTwoPhaseMixture mixture(U, phi);

//non-wetting
autoPtr<incompressiblePhase> phasea = incompressiblePhase::New(mesh,transportProperties,"a");
volVectorField& Ua = phasea->U();
surfaceScalarField& phia = phasea->phi();
const dimensionedScalar& rhoa = phasea->rho();
const dimensionedScalar& mua = phasea->mu();

// wetting
autoPtr<incompressiblePhase> phaseb = incompressiblePhase::New(mesh,transportProperties,"b");
volVectorField& Ub = phaseb->U();
surfaceScalarField& phib = phaseb->phi();
const dimensionedScalar& rhob = phaseb->rho();
const dimensionedScalar& mub = phaseb->mu();

/////////////////////////////////////////////////////////////////////////////
////////////////////////// POROUS MEDIA PROPERTIES //////////////////////////
/////////////////////////////////////////////////////////////////////////////

unsaturatedPorousModel porousMedia(mesh,transportProperties,Sb,phasea,phaseb);

// Porosity
Info << nl << "Reading porosity field eps" << endl;
const volScalarField & eps = porousMedia.porosity();


Info << nl << "Reading mobilities" << endl;
// absolute permeability is included in mobilities

const surfaceScalarField &Mbf = porousMedia.Mbf();
const surfaceScalarField &Lbf = porousMedia.Lbf();

const surfaceScalarField &Mf = porousMedia.Mf();
const surfaceScalarField &Lf = porousMedia.Lf();




/////////////////////////////////////////////////////////////////////////////
////////////////////////// VELOCITY - FLUXES ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
Info << nl << "Reading field U" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    Ua + Ub
);

#include "createPhi.H"
surfaceScalarField phiP = phi;

//U= Ua+Ub;
///////////////////////////////////////////////////////////////////
////////////////////////// FORCING TERMS //////////////////////////
///////////////////////////////////////////////////////////////////

volScalarField sourceTerm
(
    IOobject
    (
        "sourceTerm",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("",dimless/dimTime,0)
);

//geochemicalModel geoChemistry(mesh,transportProperties);

///////////////////////////////////////////////////////////////////
////////////////////////// CHEMISTRY //////////////////////////
///////////////////////////////////////////////////////////////////

IOdictionary chemistryDict
(
    IOobject
    (
        "chemistry",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

// Read Species list
wordList speciesNames(chemistryDict.lookup("species"));

// Read K-values and ZCI
List<scalar> K_values(chemistryDict.lookup("K_values"));
if (K_values.size() != speciesNames.size())
{
    FatalErrorInFunction
        << "Number of K_values does not match number of species." << nl
        << "species: " << speciesNames.size() << ", K_values: " << K_values.size() << exit(FatalError);
}

ZCI_global = List<scalar>(chemistryDict.lookup("ZCI"));

// Read T, p, R as scalars
T_global = readScalar(chemistryDict.lookup("T"));
p_global = readScalar(chemistryDict.lookup("p"));
R_global = readScalar(chemistryDict.lookup("R"));

// Optional: Check sizes match
if (ZCI_global.size() != speciesNames.size())
{
    FatalErrorInFunction
        << "Size mismatch: ZCI (" << ZCI_global.size()
        << ") vs speciesNames (" << speciesNames.size() << ")" << exit(FatalError);
}

// You can print out for debugging:
Info << "Read ZCI: " << ZCI_global << nl;
Info << "Read T: " << T_global << ", p: " << p_global << ", R: " << R_global << nl;

// Create X and Y fields
PtrList<volScalarField> Xfields(speciesNames.size());
PtrList<volScalarField> Yfields(speciesNames.size());

forAll(speciesNames, i)
{
    const word& sp = speciesNames[i];
    word Xname = "X." + sp;
    word Yname = "Y." + sp;

    Xfields.set
    (
        i,
        new volScalarField
        (
            IOobject
            (
                Xname,
                runTime.timeName(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(Xname, dimless, 0.0)
        )
    );
    Yfields.set
    (
        i,
        new volScalarField
        (
            IOobject
            (
                Yname,
                runTime.timeName(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(Yname, dimless, 0.0)
        )
    );
}

// initialize X and Y fields
const dictionary& initialMoleFractions = chemistryDict.subDict("initialMoleFractions");
List<scalar> Xa(initialMoleFractions.lookup("phasea"));
List<scalar> Xb(initialMoleFractions.lookup("phaseb"));

if (Xa.size() != speciesNames.size() || Xb.size() != speciesNames.size())
{
    FatalErrorInFunction
        << "Number of entries in initialMoleFractions does not match number of species." << nl
        << "species: " << speciesNames.size()
        << ", phasea: " << Xa.size() << ", phaseb: " << Xb.size() << exit(FatalError);
}

// Normalize
scalar sumA = 0, sumB = 0;
forAll(Xa, i) sumA += Xa[i];
forAll(Xb, i) sumB += Xb[i];
forAll(Xa, i) Xa[i] /= sumA;
forAll(Xb, i) Xb[i] /= sumB;

// Assign to fields (uniform in all cells)
forAll(speciesNames, i)
{
    Xfields[i] == Xa[i]; // assign to all cells
    Yfields[i] == Xb[i];
}
forAll(speciesNames, i)
{
    Xfields[i].correctBoundaryConditions();
    Yfields[i].correctBoundaryConditions();
}

