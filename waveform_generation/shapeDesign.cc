
#include "shapeDesign.h"
#include <random>
#include <vector>

static void verifyWholeShapeSlewRate(const char *shapeName)
{
    int nSamples = ParxRelsParGetDim("TestShapeVec", 1);
    if (nSamples < 2)
    {
        return;
    }

    double gradSampl = GradRes / 1e3;                 // s
    double maxGrad = CFG_MaxGradientStrength() / 1e3; // T/m
    double maxSlew = 0.0;

    for (int i = 1; i < nSamples; i++)
    {
        double slew = (TestShapeVec[i] - TestShapeVec[i - 1]) / gradSampl * TestShapeAmplitude / 1e2 * maxGrad;
        if (std::abs(slew) > maxSlew)
        {
            maxSlew = std::abs(slew);
        }
    }

    double maxSlewLim = CFG_GradientRampTime() / 1e3; // s
    maxSlewLim = maxGrad / maxSlewLim * 1.2;          // T/m/s

    std::cout << shapeName << " max slewrate " << maxSlew << std::endl;
    if (maxSlew > maxSlewLim)
    {
        UT_ReportError("Generated shape exceeds configured slew-rate limit");
    }
}

void rectifyTestShape(int nPre, int nPost)
{
    double gradSampl = GradRes / 1e3; // s

    std::vector<double> tmpVec;

    int shapeLen;

    shapeLen = ParxRelsParGetDim("TestShapeVec", 1);

    tmpVec.resize(shapeLen);

    for (int i = 0; i < shapeLen; i++)
    {
        tmpVec[i] = TestShapeVec[i];
    }

    if (tmpVec[0] != 0)
    {
        UT_ReportError("Test Shape not starting with zero");
    }
    bool makeRamp = false;
    int rampLen = 0;
    if (tmpVec[shapeLen - 1] != 0)
    {
        makeRamp = true;
        rampLen = int(CFG_GradientRiseTime() / GradRes);
        nPost += rampLen;
    }
    int totLen = shapeLen + nPre + nPost;

    ParxRelsParChangeDims("TestShapeVec", {totLen});

    for (int i = 0; i < totLen; i++)
    {
        if (i < nPre)
        {
            TestShapeVec[i] = 0;
        }
        else if (i >= nPre && i < nPre + shapeLen)
        {
            TestShapeVec[i] = tmpVec[i - nPre];
        }
    }
    if (makeRamp)
    {
        MRT_MakeRamp(TestShapeVec, totLen, totLen * GradRes, double(nPre + shapeLen) * GradRes, double(nPre + shapeLen) * GradRes, tmpVec[shapeLen - 1], 0, ramp_lin);
    }
    MRT_MakeRamp(TestShapeVec, totLen, totLen * GradRes, double(nPre + shapeLen + rampLen) * GradRes, double(totLen) * GradRes, 0, 0, ramp_lin);

    double _dw = 1000 / (PVM_EffSWh);
    ReadDur = totLen * gradSampl;
    AcqPoints = int(ReadDur * 1000 / _dw + 1);
}

void DesignTriangle(void)
{
    ParxRelsParMakeEditable({"TestShape_NEcho", "TestShape_InterEchoTime"});
    double riseT = CFG_GradientRiseTime() * RiseTimeMult * 1e-3; // s

    double gradSampl = GradRes / 1e3;                       // s
    SingleShapeDur = riseT * 2 * TestShapeAmplitude * 1e-2; // s
    double completeDur = (SingleShapeDur + TestShape_InterEchoTime * 1e-6 + 4e-6) * TestShape_NEcho;

    int nSamples = int((completeDur) / gradSampl + 1);

    ParxRelsParChangeDims("TestShapeVec", {nSamples});
    ParxRelsParChangeDims("GradAmpChirp", {1});

    MRT_MakeRamp(TestShapeVec, nSamples, nSamples * gradSampl, 0, nSamples * gradSampl, 0, 0, ramp_lin);

    GradAmpChirp[0] = 1.0;
    int i;

    double tAmpMult = 1.0;
    double db, d0, d1, d2, d3;
    db = 0;
    d0 = 0;
    d3 = 0;
    for (i = 0; i < TestShape_NEcho; i++)
    {
        db += (d3 - db);
        d0 = db + 4e-6;
        d1 = d0 + riseT * TestShapeAmplitude * 1e-2;
        d2 = d1 + riseT * TestShapeAmplitude * 1e-2;
        d3 = d2 + TestShape_InterEchoTime * 1e-6;

        if (AlternateMultiEcho)
        {
            if (TestShapeAmplitude > 50)
            {
                tAmpMult = ((i) % 2 == 0) ? 1 : -1;
            }
            else
            {
                tAmpMult = ((i / AlternateMultiPeriod) % 2 == 0) ? 1 : -1;
            }
        }

        MRT_MakeRamp(TestShapeVec, nSamples, nSamples * gradSampl, db, d0, 0, 0, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamples, nSamples * gradSampl, d0, d1, 0, tAmpMult, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamples, nSamples * gradSampl, d1, d2, tAmpMult, 0, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamples, nSamples * gradSampl, d2, d3, 0, 0, ramp_lin);
    }

    double _dw = 1 / (PVM_EffSWh); // s

    ReadDur = nSamples * gradSampl; // s
    TestShapeDur = ReadDur * 1e3;   // ms

    AcqPoints = int(ReadDur / _dw + 1);

    rectifyTestShape();
}

void DesignSpiral(void)
{

    double maxGrad = CFG_MaxGradientStrength() / 1e3; // T/m
    double maxSlew = CFG_GradientRampTime() / 1e3;    // T/m/s

    maxSlew = maxGrad / maxSlew * 1.2; // T/m/s

    double gamma = 42.5764 * CFG_GammaRatio(PVM_Nucleus1) * 2 * M_PI * 1 * 1e6;

    int nSample = 64;
    double fov = 80e-3;
    auto g = spiral_gradient(fov, fov / double(nSample), GradRes * 1e-3, maxSlew, maxGrad, gamma);

    TestShapeAmplitude = 95;
    int nSamples = int(g.size() + 2);

    ParxRelsParChangeDims("TestShapeVec", {nSamples});
    TestShapeDur = nSamples * GradRes;
    for (int i = 0; i < nSamples; i++)
    {
        if (i < 2)
        {
            TestShapeVec[i] = 0;
            continue;
        }

        TestShapeVec[i] = g[i - 2][0] / maxGrad;
    }

    double gradSampl = GradRes / 1e3; // s
    double _dw = 1000 / (PVM_EffSWh);
    ReadDur = (nSamples)*gradSampl;

    SingleShapeDur = ReadDur; // s

    AcqPoints = int(ReadDur * 1000 / _dw + 1);
    rectifyTestShape();
}

void DesignTrapz(void)
{

    ParxRelsParMakeEditable({"TestShapeDur"});

    double gradSampl = GradRes / 1e3; // s

    double T = TestShapeDur * 1e-3; // seconds

    int nSamplesShape = int((T) / gradSampl) / 2;
    int nSamples = int((T) / gradSampl);

    ParxRelsParChangeDims("TestShapeVec", {nSamples});
    ParxRelsParChangeDims("GradAmpChirp", {1});
    GradAmpChirp[0] = 1.0;

    int i;
    for (i = 0; i < nSamples; i++)
    {

        if (i < nSamples / 2)
        {
            TestShapeVec[i] = 2 * (-(std::abs(double(i) / double(nSamplesShape) - 0.5)) + 0.5);
        }
        else
        {
            TestShapeVec[i] = -2 * (-(std::abs(double(i - nSamplesShape) / double(nSamplesShape) - 0.5)) + 0.5);
        }
    }

    double _dw = 1000 / (PVM_EffSWh);
    ReadDur = nSamples * gradSampl;
    AcqPoints = int(ReadDur * 1000 / _dw + 1);

    SingleShapeDur = ReadDur; // s
    rectifyTestShape();
}

void DesignRamp(void)
{

    double gradSampl = GradRes * 1e-3; // s

    double riseT = CFG_GradientRiseTime(); // ms

    TestShapeDur = 40;
    double T = TestShapeDur * 1e-3; // seconds

    int nSamplesShape = int((T) / gradSampl);
    int nSamples = int((T) / gradSampl);

    ParxRelsParChangeDims("TestShapeVec", {nSamples});
    ParxRelsParChangeDims("GradAmpChirp", {1});

    double totalTimeMs = nSamplesShape * GradRes;
    MRT_MakeRamp(TestShapeVec, nSamplesShape, totalTimeMs, 0, totalTimeMs, 0, 0, ramp_lin);

    GradAmpChirp[0] = 1.0;

    double d1, d2, d3, d4;

    d1 = 2 * GradRes;
    d2 = d1 + riseT;
    d3 = d2 + 2;
    d4 = d3 + riseT;

    MRT_MakeRamp(TestShapeVec, nSamplesShape, totalTimeMs, 0, d1, 0, 0, ramp_lin);
    MRT_MakeRamp(TestShapeVec, nSamplesShape, totalTimeMs, d1, d2, 0, 1, ramp_lin);
    MRT_MakeRamp(TestShapeVec, nSamplesShape, totalTimeMs, d2, d3, 1, 1, ramp_lin);
    MRT_MakeRamp(TestShapeVec, nSamplesShape, totalTimeMs, d3, d4, 1, 0, ramp_lin);
    MRT_MakeRamp(TestShapeVec, nSamplesShape, totalTimeMs, d4, totalTimeMs, 0, 0, ramp_lin);

    double _dw = 1000 / (PVM_EffSWh);
    ReadDur = nSamples * gradSampl;
    AcqPoints = int(ReadDur * 1000 / _dw + 1);
    SingleShapeDur = ReadDur; // s
    rectifyTestShape();
}

void DesignRose(void)
{
    ParxRelsParMakeEditable({"TestShapeDur"});

    double gradSampl = GradRes / 1e3;                 // s
    double maxGrad = CFG_MaxGradientStrength() / 1e3; // T/m
    // double rampTime = CFG_GradientRampTime() / 1e3;                         // T/m/s

    double T = TestShapeDur * 1e-3; // seconds
    // double maxSlew = CFG_GradientRampTime() / 1e3;                         // T/m/s

    int nSamples = int((T) / gradSampl + 0.5);

    // maxSlew = TestShapeAmplitude

    ParxRelsParChangeDims("TestShapeVec", {nSamples});
    ParxRelsParChangeDims("GradAmpChirp", {1});
    GradAmpChirp[0] = 1.0;
    std::vector<double> dif;
    dif.resize(nSamples);
    int i = 0;
    double phi;
    double maxSlew = 0;
    double n, d, k;

    n = 3.0;
    d = 4.0;
    k = n / d;

    for (i = 0; i < nSamples; i++)
    {
        TestShapeVec[i] = 0;
        dif[i] = 0;
    }

    for (i = 0; i < nSamples; i++)
    {
        phi = double(i) / double(nSamples) * 2.0 * M_PI * d;
        TestShapeVec[i] = -std::sin(phi) * std::cos(k * phi) - k * std::sin(k * phi) * std::cos(phi);
        if (i != 0)
        {
            dif[i] = (TestShapeVec[i] - TestShapeVec[i - 1]) / gradSampl * TestShapeAmplitude / 1e2 * maxGrad;
        }
        else
        {
            dif[i] = 0;
        }
        if (std::abs(dif[i]) > maxSlew)
        {
            maxSlew = std::abs(dif[i]);
        }
    }

    double maxSlewLim = CFG_GradientRampTime() / 1e3; // T/m/s
    maxSlewLim = maxGrad / maxSlewLim * 1.2;          // T/m/s

    std::cout << "Max slewrate" << maxSlew << std::endl;

    if (maxSlew > maxSlewLim)
    {
        TestShapeAmplitude *= maxSlewLim / maxSlew;
        std::cout << "Compensating slewrate" << TestShapeAmplitude << std::endl;
    }

    double _dw = 1000 / (PVM_EffSWh);
    ReadDur = nSamples * gradSampl;
    AcqPoints = int(ReadDur * 1000 / _dw + 1);
    SingleShapeDur = ReadDur; // s
    rectifyTestShape();
}

void DesignPrgw(void)
{
    // PRGW builds a pseudo-random gradient waveform from slew-limited level changes.
    // PrbsSeed makes the random levels reproducible; EpiFlattop sets the minimum
    // number of samples to hold each target level before choosing a new one. While
    // walking through the samples, cumulativeArea tracks the signed gradient area
    // in normalized sample units so the generator can keep net dephasing bounded
    // and steer the tail back toward zero before rectifyTestShape() pads/finishes it.
    ParxRelsParMakeEditable({"TestShapeDur", "PrbsSeed", "EpiFlattop"});

    double gradSampl = GradRes / 1e3;                                                 // s
    double riseT = CFG_GradientRiseTime() * RiseTimeMult * TestShapeAmplitude * 1e-2; // ms, 0 to requested full scale
    double T = TestShapeDur * 1e-3;                                                   // seconds

    int nSamples = int(T / gradSampl + 0.5);
    nSamples = MAX_OF(nSamples, 2);

    int riseSamples = int(riseT / GradRes + 0.5);
    riseSamples = MAX_OF(riseSamples, 1);

    const int minFlatTopSamples = MAX_OF(int(EpiFlattop * 1e-3 / GradRes + 0.999999), 1);
    int minBlockSamples = minFlatTopSamples;
    int maxBlockSamples = MAX_OF(2 * riseSamples + minBlockSamples, minBlockSamples);
    int correctionSamples = MAX_OF(4 * riseSamples, 8);
    double maxStep = 1.0 / double(riseSamples);

    ParxRelsParChangeDims("TestShapeVec", {nSamples});
    ParxRelsParChangeDims("GradAmpChirp", {1});
    GradAmpChirp[0] = 1.0;

    std::mt19937 rng((unsigned int)PrbsSeed);
    std::uniform_real_distribution<double> levelDistribution(-1.0, 1.0);
    std::uniform_int_distribution<int> flatTopDistribution(minBlockSamples, maxBlockSamples);
    const double maxPrbsDephasing = 85.0; // percent signal loss cap derived from cumulative PRGW dephasing
    double target = 0.0;
    double current = 0.0;
    double cumulativeArea = 0.0;
    double maxCumulativeArea = 1e30;
    // Reserve the end of the waveform for returning the gradient to zero. The
    // correction window just before that uses the remaining samples to cancel
    // accumulated signed area as much as the slew limit allows.
    int tailStart = MAX_OF(nSamples - riseSamples, 1);
    int correctionStart = MAX_OF(tailStart - correctionSamples, 1);
    int nextTargetSample = 1;

    if (maxPrbsDephasing < 100.0 && TestShapeAmplitude > 0.0)
    {
        // Convert the desired signal-loss cap into a maximum allowed cumulative
        // normalized area. The physical area per sample depends on GradRes,
        // gradient calibration, and the requested percent amplitude.
        double sigma = PVM_SliceThick / 2.35482;
        double remainingSignal = 1.0 - maxPrbsDephasing / 100.0;
        remainingSignal = MAX_OF(remainingSignal, 1e-6);
        double maxPhysicalArea = sqrt(-log(remainingSignal) / (2.0 * M_PI * M_PI * sigma * sigma));
        double areaPerSample = GradRes * 2e-3 * PVM_GradCalConst * 1e-2 * TestShapeAmplitude;
        if (areaPerSample > 0.0)
        {
            maxCumulativeArea = maxPhysicalArea / areaPerSample;
        }
    }

    for (int i = 0; i < nSamples; i++)
    {
        if (i == 0)
        {
            TestShapeVec[i] = 0.0;
            continue;
        }

        if (i >= tailStart)
        {
            target = 0.0;
        }
        else if (i >= correctionStart)
        {
            int remainingSamples = MAX_OF(tailStart - i, 1);
            target = -cumulativeArea / double(remainingSamples);
            if (target > 1.0)
            {
                target = 1.0;
            }
            else if (target < -1.0)
            {
                target = -1.0;
            }
        }
        else if (i >= nextTargetSample)
        {
            // Pick one pseudo-random level and hold it for a random flat-top after
            // the slew-limited ramp needed to reach that level.
            target = levelDistribution(rng);
            int flatTopSamples = flatTopDistribution(rng);
            int rampSamples = int(std::abs(target - current) / maxStep + 0.999999);
            int blockSamples = rampSamples + flatTopSamples;
            blockSamples = MIN_OF(blockSamples, correctionStart - i);
            blockSamples = MAX_OF(blockSamples, 1);
            nextTargetSample = i + blockSamples;
        }

        double proposedCurrent = current;
        double delta = target - current;
        if (delta > maxStep)
        {
            delta = maxStep;
        }
        else if (delta < -maxStep)
        {
            delta = -maxStep;
        }
        proposedCurrent += delta;

        double slewMinCurrent = current - maxStep;
        double slewMaxCurrent = current + maxStep;
        double areaMinCurrent = -maxCumulativeArea - cumulativeArea;
        double areaMaxCurrent = maxCumulativeArea - cumulativeArea;
        double allowedMinCurrent = MAX_OF(MAX_OF(slewMinCurrent, areaMinCurrent), -1.0);
        double allowedMaxCurrent = MIN_OF(MIN_OF(slewMaxCurrent, areaMaxCurrent), 1.0);

        // Clamp each output sample to the intersection of hardware-like slew,
        // dephasing/area, and normalized amplitude limits. Otherwise keep it as is 
        // to best track the random target and preserve the desired statistics of the PRGW.
        if (allowedMinCurrent <= allowedMaxCurrent)
        {
            if (proposedCurrent > allowedMaxCurrent)
            {
                proposedCurrent = allowedMaxCurrent;
            }
            else if (proposedCurrent < allowedMinCurrent)
            {
                proposedCurrent = allowedMinCurrent;
            }
        }
        else
        {
            // If the dephasing cap becomes unreachable in one sample, preserve slew-limited motion
            // and bias toward reducing the accumulated area as fast as hardware allows.
            proposedCurrent = (cumulativeArea >= 0.0) ? slewMinCurrent : slewMaxCurrent;
            if (proposedCurrent > 1.0)
            {
                proposedCurrent = 1.0;
            }
            else if (proposedCurrent < -1.0)
            {
                proposedCurrent = -1.0;
            }
        }

        current = proposedCurrent;
        TestShapeVec[i] = current;
        cumulativeArea += current;
    }

    double _dw = 1000 / (PVM_EffSWh);
    ReadDur = nSamples * gradSampl;
    AcqPoints = int(ReadDur * 1000 / _dw + 1);
    SingleShapeDur = ReadDur; // s

    rectifyTestShape();
    verifyWholeShapeSlewRate("PRGW");
}

void CalculateDephasingGradient(double areaToDephase, double &dephaseDur, double &dephaseAmp, double minPlateauSamples, double riseT)
{

    // calculations based on the quadratic equation  derived from the area of the trapezoid (minPlateauSamples + trueRiseT)*dephGradVal = limitingArea
    // trueRiseT  =  riseT * dephGradVal/1e2 ... ensures constant and maximum slew rate
    // then solve for readGradVal

    double readDephGradLim = 80;

    double maxTriangleArea = riseT * readDephGradLim + minPlateauSamples * GradRes * readDephGradLim; // is in fact maximum trapezoid area with minPlateauSamples in the upper part
    if (maxTriangleArea >= areaToDephase)
    {
        std::cout << "ReadOut dephasing not prolonged........." << std::endl;
        dephaseAmp = (-minPlateauSamples * GradRes + sqrt(minPlateauSamples * minPlateauSamples * GradRes * GradRes + 4 * areaToDephase * riseT / 1e2)) / (2 * riseT / 1e2);
        double trueRiseTime = round((riseT * dephaseAmp / 1e2) / GradRes + 0.5) * GradRes;
        dephaseDur = (minPlateauSamples)*GradRes + trueRiseTime;
        dephaseAmp = areaToDephase / (dephaseDur);
    }
    else
    {
        std::cout << "Prolonging dephasing..........." << std::endl;
        double AreaToComp = areaToDephase - riseT * readDephGradLim;
        dephaseAmp = readDephGradLim;
        dephaseDur = riseT + AreaToComp / (readDephGradLim);
    }
}

void DesignReadout(void)
{
    double readOutRiseT, maxEncGrad, acqEchoTime;

    ParxRelsParMakeEditable({"ReadPoints", "FlattopSamples", "TestShape_NEcho", "TestShape_InterEchoTime"});

    double gradSampl = GradRes; // s

    double riseT = CFG_GradientRiseTime() * RiseTimeMult;

    maxEncGrad = TestShapeAmplitude;

    std::cout << "Max Enc grad" << maxEncGrad << std::endl;

    readOutRiseT = riseT * maxEncGrad / 100.0;

    readOutRiseT /= GradRes;
    readOutRiseT += 1.5;
    readOutRiseT = round(readOutRiseT) * GradRes;

    acqEchoTime = 1000 / PVM_EffSWh * ReadPoints * 0.5;

    double areaToDefocusRead = acqEchoTime * maxEncGrad + readOutRiseT * maxEncGrad / 2; // area of gradient to echo position

    double dephGradVal, dephGradDur;

    CalculateDephasingGradient(areaToDefocusRead, dephGradDur, dephGradVal, (double)FlattopSamples, riseT);

    double intRiseT = riseT * std::abs(dephGradVal) * 1e-2;
    double d0 = 2 * GradRes;
    double d1 = intRiseT + d0;
    double d2 = d1 + dephGradDur - intRiseT;
    double d3 = d2 + intRiseT + readOutRiseT;
    double d4 = d3 + 1000 / PVM_EffSWh * ReadPoints;
    double d5 = d4 + readOutRiseT;
    double d6 = d5 + TestShape_InterEchoTime * 1e-3;
    SingleShapeDur = d6;

    double totalShapeDur = (d5 + TestShape_InterEchoTime * 1e-3) * TestShape_NEcho;

    int nSamplesShape = int((totalShapeDur) / gradSampl + 0.5);

    ParxRelsParChangeDims("TestShapeVec", {nSamplesShape});

    double _dw = 1000 / (PVM_EffSWh);

    ReadDur = nSamplesShape * gradSampl / 1e3;

    AcqPoints = int(ReadDur * 1000 / _dw + 1);

    for (int i = 0; i < nSamplesShape; i++)
    {
        TestShapeVec[i] = 0;
    }
    double ampPolarity = 1;
    double tStart = 0;
    for (int i = 0; i < TestShape_NEcho; i++)
    {
        if (AlternateMultiEcho)
        {
            ampPolarity = ((i / AlternateMultiPeriod) % 2 == 0) ? 1 : -1;
        }
        tStart = double(i) * SingleShapeDur;

        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, tStart, d0 + tStart, 0, 0, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d0 + tStart, d1 + tStart, 0, -ampPolarity * dephGradVal / maxEncGrad, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d1 + tStart, d2 + tStart, -ampPolarity * dephGradVal / maxEncGrad, -ampPolarity * dephGradVal / maxEncGrad, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d2 + tStart, d3 + tStart, -ampPolarity * dephGradVal / maxEncGrad, ampPolarity * 1, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d3 + tStart, d4 + tStart, ampPolarity * 1, ampPolarity * 1, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d4 + tStart, d5 + tStart, ampPolarity * 1, 0, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d5 + tStart, d6 + tStart, 0, 0, ramp_lin);
    }

    ParxRelsParChangeDims("GradAmpChirp", {1});
    TestShapeDur = totalShapeDur;
    GradAmpChirp[0] = 1.0;

    rectifyTestShape();
}

void DesignMge(void)
{
    double readOutRiseT, maxEncGrad, acqEchoTime;

    int nEcho = TestShape_NEcho;

    ParxRelsParMakeEditable({"ReadPoints", "FlattopSamples", "TestShape_InterEchoTime", "TestShape_NEcho"});

    double gradSampl = GradRes; // s

    double riseT = CFG_GradientRiseTime() * RiseTimeMult;

    maxEncGrad = TestShapeAmplitude;

    std::cout << "Max Enc grad" << maxEncGrad << std::endl;

    readOutRiseT = riseT * maxEncGrad / 100.0;

    readOutRiseT /= GradRes;
    readOutRiseT += 1.5;
    readOutRiseT = round(readOutRiseT) * GradRes;

    acqEchoTime = 1000 / PVM_EffSWh * ReadPoints * 0.5;

    double areaToDefocusRead = (acqEchoTime * maxEncGrad + readOutRiseT * maxEncGrad / 2) * 2; // area of gradient to echo position

    double dephGradVal, dephGradDur;

    CalculateDephasingGradient(areaToDefocusRead, dephGradDur, dephGradVal, (double)FlattopSamples, riseT);

    double intRiseT = riseT * std::abs(dephGradVal) * 1e-2;

    double dbase = 0;
    double d0 = TestShape_InterEchoTime * 1e-3;
    double d1 = intRiseT + d0;

    double d2 = d1 + dephGradDur - intRiseT;

    double d3 = d2 + intRiseT + readOutRiseT;

    double d4 = d3 + 1000 / PVM_EffSWh * ReadPoints;
    double d5 = d4 + readOutRiseT;

    int nSamplesShape = int((d5) / gradSampl + 0.5);

    ParxRelsParChangeDims("TestShapeVec", {nSamplesShape * nEcho});

    double _dw = 1000 / (PVM_EffSWh);

    ReadDur = nEcho * nSamplesShape * gradSampl / 1e3;
    SingleShapeDur = ReadDur / nEcho; // s

    AcqPoints = int(ReadDur * 1000 / _dw + 1);

    for (int i = 0; i < nSamplesShape * nEcho; i++)
    {
        TestShapeVec[i] = 0;
    }
    double dephaseAmplitude = -dephGradVal / maxEncGrad;
    for (int i = 0; i < nEcho; i++)
    {
        dbase = double(i) * (nSamplesShape * gradSampl);
        d0 = TestShape_InterEchoTime * 1e-3 + dbase;
        d1 = intRiseT + d0;

        d2 = d1 + dephGradDur - intRiseT;

        d3 = d2 + intRiseT + readOutRiseT;

        d4 = d3 + 1000 / PVM_EffSWh * ReadPoints;
        d5 = d4 + readOutRiseT;

        if (i == 0)
        {
            dephaseAmplitude = (-dephGradVal / maxEncGrad) / 2;
        }
        else
        {
            dephaseAmplitude = -dephGradVal / maxEncGrad;
        }

        MRT_MakeRamp(TestShapeVec, nSamplesShape * nEcho, ReadDur * 1e3, dbase, d0, 0, 0, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape * nEcho, ReadDur * 1e3, d0, d1, 0, dephaseAmplitude, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape * nEcho, ReadDur * 1e3, d1, d2, dephaseAmplitude, dephaseAmplitude, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape * nEcho, ReadDur * 1e3, d2, d3, dephaseAmplitude, 1, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape * nEcho, ReadDur * 1e3, d3, d4, 1, 1, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape * nEcho, ReadDur * 1e3, d4, d5, 1, 0, ramp_lin);
    }

    ParxRelsParChangeDims("GradAmpChirp", {1});
    TestShapeDur = d5;
    GradAmpChirp[0] = 1.0;

    rectifyTestShape();
}

void DesignEPI(void)
{

    ParxRelsParMakeEditable({"EpiFlattop", "TestShape_NEcho"});

    double gradSampl = GradRes; // s
    double riseT = CFG_GradientRiseTime() * TestShapeAmplitude * 1e-2 * RiseTimeMult;

    double epiFlattop = EpiFlattop * 1e-3;

    double d1 = riseT;
    double d2 = d1 + epiFlattop;
    double d3 = d2 + riseT;

    int nReads = TestShape_NEcho;
    int nSamplesShape = int((nReads * d3 + 2 * GradRes) / gradSampl + 0.5);
    ParxRelsParChangeDims("TestShapeVec", {nSamplesShape});
    for (int i = 0; i < nSamplesShape; i++)
    {
        TestShapeVec[i] = 0;
    }

    double _dw = 1000 / (PVM_EffSWh);
    ReadDur = nSamplesShape * gradSampl / 1e3;
    AcqPoints = int(ReadDur * 1000 / _dw + 1);

    double tOffset;
    for (int i = 0; i < nReads; i++)
    {
        tOffset = d3 * double(i) + 2 * GradRes;

        double amp = pow(-1, i);

        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, 0 + tOffset, d1 + tOffset, 0, amp, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d1 + tOffset, d2 + tOffset, amp, amp, ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d2 + tOffset, d3 + tOffset, amp, 0, ramp_lin);
    }

    MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d3 + tOffset, ReadDur * 1e3, 0, 0, ramp_lin);

    ParxRelsParChangeDims("GradAmpChirp", {1});
    GradAmpChirp[0] = 1.0;
    TestShapeDur = ReadDur * 1000;
    SingleShapeDur = ReadDur / nReads; // s
    rectifyTestShape();
}

void DesignTRAPZ_SERIES(void)
{

    ParxRelsParMakeEditable({"EpiFlattop", "TestShape_NEcho"});

    double gradSampl = GradRes; // s
    double riseT = CFG_GradientRiseTime() * RiseTimeMult;

    int nTrapzs = 9;

    double spacing = 0.2;

    double trapzTop = EpiFlattop * 1e-3;

    double amps[nTrapzs] = {-1.0, 1.0, 0.9, -0.8, -0.7, 0.6, 0.5, -0.3, -0.2};

    double maxriseT = riseT * TestShapeAmplitude * 1e-2;

    double shapeDur = spacing * (nTrapzs - 1) + (trapzTop + 2 * maxriseT) * nTrapzs + 2 * GradRes;

    int nSamplesShape = int(shapeDur / gradSampl + 0.5);

    ParxRelsParChangeDims("TestShapeVec", {nSamplesShape});

    for (int i = 0; i < nSamplesShape; i++)
    {
        TestShapeVec[i] = 0.0;
    }

    double _dw = 1000 / (PVM_EffSWh);
    ReadDur = nSamplesShape * gradSampl / 1e3;
    SingleShapeDur = ReadDur / nTrapzs; // s
    AcqPoints = int(ReadDur * 1000 / _dw + 1);

    double d0, d1, d2, d3, move = 0.0;

    move = 2 * GradRes;

    for (int i = 0; i < nTrapzs; i++)
    {
        double intRiseT = riseT * std::abs(amps[i]) * TestShapeAmplitude * 1e-2;
        d0 = move;
        d1 = d0 + intRiseT;
        d2 = d1 + trapzTop;
        d3 = d2 + intRiseT;
        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d0, d1, 0, amps[i], ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d1, d2, amps[i], amps[i], ramp_lin);
        MRT_MakeRamp(TestShapeVec, nSamplesShape, ReadDur * 1e3, d2, d3, amps[i], 0, ramp_lin);

        move += d3 - d0 + spacing;
    }

    ParxRelsParChangeDims("GradAmpChirp", {1});
    GradAmpChirp[0] = 1.0;
    TestShapeDur = ReadDur * 1000;

    rectifyTestShape();
}

void DesignChirp(void)
{
    ParxRelsParMakeEditable({"ChirpFmin",
                             "ChirpFmax",
                             "TestShapeDur"});

    ChirpFmin = MIN_OF(ChirpFmin, ChirpFmax - 1);

    double gradSampl = GradRes / 1e3;                 // s
    double maxGrad = CFG_MaxGradientStrength() / 1e3; // T/m
    double maxSlew = CFG_GradientRampTime() / 1e3;    // T/m/s

    maxSlew = maxGrad / maxSlew; // T/m/s

    std::cout << "Max Slewrate:" << maxSlew << std::endl;

    double freqmin = ChirpFmin * 1000; // kHz
    double freqmax = ChirpFmax * 1000; // kHz
    double gradMax[3] = {50, 50, 50};  // %

    gradMax[0] = TestShapeAmplitude;
    gradMax[1] = TestShapeAmplitude;
    gradMax[2] = TestShapeAmplitude;

    TestShapeDur = MAX_OF(TestShapeDur, 5);
    double T = TestShapeDur * 1e-3; // seconds

    double c = (freqmax - freqmin) / T;

    int nSamples = int((T) / gradSampl);

    ParxRelsParChangeDims("TestShapeVec", {nSamples});
    ParxRelsParChangeDims("GradAmpChirp", {1});
    GradAmpChirp[0] = 1.0;

    for (int i = 0; i < nSamples; i++)
    {
        double t = i * gradSampl;
        double normValue = MAX_OF(maxSlew, gradMax[0] / 100.0 * maxGrad * 2 * M_PI * (c * t + freqmin));

        TestShapeVec[i] = gradMax[0] / 100.0 * maxGrad * sin(2 * M_PI * (c / 2 * t * t + freqmin * t)) / normValue * maxSlew;
    }

    double integral = 0;

    for (int i = 0; i < nSamples; i++)
    {
        integral += TestShapeVec[i];
    }

    integral *= gradSampl; // To calculate prephasing

    // calculate prehasing from a single lobe of sine function with defined frequency and amplitude
    // Conditions: 1st derivative same as the beginning of TestShapeVec (meeting point)
    //             2nd Area to compensate chirp integral (to ensure we wobble around kspace center)

    double normValueDer = MAX_OF(maxSlew, gradMax[0] / 100.0 * maxGrad * 2 * M_PI * (freqmin));

    double der_0 = gradMax[0] / 100.0 * maxGrad * 2 * M_PI * (freqmin) / normValueDer * maxSlew; // derivative at chirp start

    double f = sqrt(der_0 / (2 * M_PI * M_PI * integral)); // Expected sin frequency

    double A = der_0 / (2 * M_PI * f); // Expected sin amplitude

    int nSamplePrepend = (int)(1 / (2 * f) / gradSampl);

    std::vector<double> prephaseShape(nSamplePrepend);

    for (int i = 0; i < nSamplePrepend; i++)
    {
        prephaseShape[i] = -A * sin(2 * M_PI * f * i * gradSampl);
    }

    // create final chirp

    ParxRelsParChangeDims("TestShapeVec", {nSamples + nSamplePrepend});

    for (int i = 0; i < (nSamples + nSamplePrepend); i++)
    {
        TestShapeVec[i] = 0;
    }

    for (int i = 0; i < (nSamples + nSamplePrepend); i++)
    {

        if ((i) < nSamplePrepend)
        {
            TestShapeVec[i] = prephaseShape[i];
        }
        else
        {
            double t = ((i)-nSamplePrepend) * gradSampl;
            double normValue = MAX_OF(maxSlew, gradMax[0] / 100.0 * maxGrad * 2 * M_PI * (c * t + freqmin));
            TestShapeVec[i] = gradMax[0] / 100.0 * maxGrad * sin(2 * M_PI * (c / 2 * t * t + freqmin * t)) / normValue * maxSlew;
        }
    }

    nSamples = nSamples + nSamplePrepend;

    for (int i = 0; i < nSamples; i++)
    {
        TestShapeVec[i] = TestShapeVec[i] / (gradMax[0] / 100.0 * maxGrad);
    }

    std::vector<double> cumSum(nSamples);
    double max = 0;
    for (int i = 0; i < nSamples; i++)
    {
        cumSum[i] = (i == 0 ? TestShapeVec[0] : cumSum[i - 1]) + TestShapeVec[i] * gradSampl;

        if (fabs(cumSum[i]) > max)
        {
            max = fabs(cumSum[i]);
        }
    }

    max = max * CFG_GradCalConst(PVM_Nucleus1);
    double _dw = 1000 / (PVM_EffSWh);
    ReadDur = (nSamples)*gradSampl;
    AcqPoints = int(ReadDur * 1000 / _dw + 1);
    SingleShapeDur = ReadDur; // s
    rectifyTestShape();
}
