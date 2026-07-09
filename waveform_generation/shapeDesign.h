#ifndef CHIRP_DESIGN_H
#define CHIRP_DESIGN_H

#include "method.h"
#include "spiral.h"

// ----- Main design functions (declared exactly as in your source) -----
void DesignTriangle(void);
void DesignSpiral(void);
void DesignTrapz(void);
void DesignRamp(void);
void DesignRose(void);
void DesignPrgw(void);
void CalculateDephasingGradient(double areaToDephase, double &dephaseDur, double &dephaseAmp, double minPlateauSamples, double riseT);
void DesignReadout(void);
void DesignEPI(void);
void DesignTRAPZ_SERIES(void);
void DesignChirp(void);
void DesignMge(void);
void rectifyTestShape(int nPre = 100, int nPost = 100);

#endif // CHIRP_DESIGN_H
