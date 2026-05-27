#include "AeroNNComponentV2.h"

#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNEModelData.h"

UAeroNNComponentV2::UAeroNNComponentV2()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UAeroNNComponentV2::BeginPlay()
{
    Super::BeginPlay();

    BuildNACASurface();

    if (!InitModel())
    {
        UE_LOG(LogTemp, Error, TEXT("AeroNNComponentV2: initialization failed"));
        return;
    }

    bReady = true;

    UE_LOG(
        LogTemp,
        Display,
        TEXT("AeroNNComponentV2: ready with %d surface samples"),
        SurfaceSamples.Num()
    );
}

bool UAeroNNComponentV2::InitModel()
{
    if (!ModelData)
    {
        UE_LOG(LogTemp, Error, TEXT("AeroNNComponentV2: ModelData is null"));
        return false;
    }

    TWeakInterfacePtr<INNERuntimeCPU> Runtime =
        UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));

    if (!Runtime.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("AeroNNComponentV2: Could not get NNERuntimeORTCpu"));
        return false;
    }

    TSharedPtr<UE::NNE::IModelCPU> Model = Runtime->CreateModelCPU(ModelData);

    if (!Model.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("AeroNNComponentV2: Could not create model"));
        return false;
    }

    ModelInstance = Model->CreateModelInstanceCPU();

    if (!ModelInstance.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("AeroNNComponentV2: Could not create model instance"));
        return false;
    }

    // V9 ONNX input is [N, 5], output [N, 1].
    return EnsureInputShape(1);
}

bool UAeroNNComponentV2::EnsureInputShape(int32 BatchSize)
{
    if (!ModelInstance.IsValid())
    {
        return false;
    }

    if (CurrentBatchSize == BatchSize)
    {
        return true;
    }

    TArray<UE::NNE::FTensorShape> InputShapes;
    InputShapes.Add(UE::NNE::FTensorShape::Make({
        static_cast<uint32>(BatchSize),
        5u
        }));
    const auto Status = ModelInstance->SetInputTensorShapes(InputShapes);

    if (Status != UE::NNE::IModelInstanceRunSync::ESetInputTensorShapesStatus::Ok)
    {
        UE_LOG(
            LogTemp,
            Error,
            TEXT("AeroNNComponentV2: Failed to set input shape [%d, 5]"),
            BatchSize
        );
        return false;
    }

    CurrentBatchSize = BatchSize;
    return true;
}

float UAeroNNComponentV2::NacaCamberY(float X01, float M, float P)
{
    if (FMath::Abs(M) < KINDA_SMALL_NUMBER)
    {
        return 0.0f;
    }

    if (X01 < P)
    {
        return M / (P * P) * (2.0f * P * X01 - X01 * X01);
    }

    return M / ((1.0f - P) * (1.0f - P))
        * ((1.0f - 2.0f * P) + 2.0f * P * X01 - X01 * X01);
}

void UAeroNNComponentV2::NacaUpperLower(float X01, float M, float P, float T, float& Yu, float& Yl)
{
    const float X = FMath::Clamp(X01, 0.0f, 1.0f);

    const float Yt = 5.0f * T * (
        0.2969f * FMath::Sqrt(X)
        - 0.1260f * X
        - 0.3516f * X * X
        + 0.2843f * X * X * X
        - 0.1015f * X * X * X * X
        );

    const float Yc = NacaCamberY(X, M, P);

    Yu = Yc + Yt;
    Yl = Yc - Yt;
}

void UAeroNNComponentV2::BuildNACASurface()
{
    SurfaceSamples.Reset();

    const int32 N = FMath::Max(16, NacaPointCount);

    TArray<FVector2D> Points;
    Points.Reserve(2 * N);

    TArray<float> X01;
    X01.Reserve(N);

    for (int32 i = 0; i < N; ++i)
    {
        const float Beta = PI * static_cast<float>(i) / static_cast<float>(N - 1);
        const float X = 0.5f * (1.0f - FMath::Cos(Beta));
        X01.Add(X);
    }

    TArray<FVector2D> Upper;
    TArray<FVector2D> Lower;

    Upper.Reserve(N);
    Lower.Reserve(N);

    for (int32 i = 0; i < N; ++i)
    {
        const float X = X01[i];

        const float Yt = 5.0f * NacaT * (
            0.2969f * FMath::Sqrt(FMath::Clamp(X, 0.0f, 1.0f))
            - 0.1260f * X
            - 0.3516f * X * X
            + 0.2843f * X * X * X
            - 0.1015f * X * X * X * X
            );

        float Yc = 0.0f;
        float Dyc = 0.0f;

        if (FMath::Abs(NacaM) > KINDA_SMALL_NUMBER)
        {
            if (X < NacaP)
            {
                Yc = NacaM / (NacaP * NacaP) * (2.0f * NacaP * X - X * X);
                Dyc = 2.0f * NacaM / (NacaP * NacaP) * (NacaP - X);
            }
            else
            {
                Yc = NacaM / ((1.0f - NacaP) * (1.0f - NacaP))
                    * ((1.0f - 2.0f * NacaP) + 2.0f * NacaP * X - X * X);

                Dyc = 2.0f * NacaM / ((1.0f - NacaP) * (1.0f - NacaP)) * (NacaP - X);
            }
        }

        const float Theta = FMath::Atan(Dyc);

        const float Xu = X - Yt * FMath::Sin(Theta);
        const float Yu = Yc + Yt * FMath::Cos(Theta);

        const float Xl = X + Yt * FMath::Sin(Theta);
        const float Yl = Yc - Yt * FMath::Cos(Theta);

        Upper.Add(FVector2D(Xu - 0.5f, Yu));
        Lower.Add(FVector2D(Xl - 0.5f, Yl));
    }

    // Same convention as Python:
    // upper LE -> TE, then lower TE -> LE.
    for (const FVector2D& P : Upper)
    {
        Points.Add(P);
    }

    for (int32 i = Lower.Num() - 1; i >= 0; --i)
    {
        Points.Add(Lower[i]);
    }

    const int32 M = Points.Num();
    if (M < 3)
    {
        return;
    }

    // Signed area.
    float Area = 0.0f;
    for (int32 i = 0; i < M; ++i)
    {
        const FVector2D& A = Points[i];
        const FVector2D& B = Points[(i + 1) % M];
        Area += A.X * B.Y - B.X * A.Y;
    }
    Area *= 0.5f;

    const int32 Half = M / 2;

    SurfaceSamples.Reserve(M);

    for (int32 i = 0; i < M; ++i)
    {
        const FVector2D& A = Points[i];
        const FVector2D& B = Points[(i + 1) % M];

        const FVector2D D = B - A;

        const float DS = FMath::Max(D.Size(), 1e-8f);
        const FVector2D Mid = 0.5f * (A + B);

        FVector2D Normal;

        if (Area < 0.0f)
        {
            Normal = FVector2D(-D.Y / DS, D.X / DS);
        }
        else
        {
            Normal = FVector2D(D.Y / DS, -D.X / DS);
        }

        FAeroSurfaceSampleV2 Sample;
        Sample.Position = Mid;
        Sample.Normal = Normal;
        Sample.DS = DS;
        Sample.Side = i < Half ? 1.0f : -1.0f;

        SurfaceSamples.Add(Sample);
    }

    UE_LOG(
        LogTemp,
        Display,
        TEXT("AeroNNComponentV2: built NACA surface with %d integration samples, signed area=%g"),
        SurfaceSamples.Num(),
        Area
    );
}

bool UAeroNNComponentV2::EvaluateCpAtPoint(
    float XPinn,
    float Y,
    float Side,
    float AlphaRad,
    float& Cp
)
{
    Cp = 0.0f;

    FAeroSurfaceSampleV2 Sample;
    Sample.Position = FVector2D(XPinn, Y);
    Sample.Normal = FVector2D::ZeroVector;
    Sample.DS = 0.0f;
    Sample.Side = Side >= 0.0f ? 1.0f : -1.0f;

    TArray<FAeroSurfaceSampleV2> Samples;
    Samples.Add(Sample);

    TArray<float> Cps;
    if (!EvaluateCpBatch(Samples, AlphaRad, Cps) || Cps.Num() != 1)
    {
        return false;
    }

    Cp = Cps[0];
    return true;
}

bool UAeroNNComponentV2::EvaluateCpBatch(
    const TArray<FAeroSurfaceSampleV2>& Samples,
    float AlphaRad,
    TArray<float>& OutCp
)
{
    OutCp.Reset();

    if (!bReady || !ModelInstance.IsValid())
    {
        return false;
    }

    const int32 BatchSize = Samples.Num();

    if (BatchSize <= 0)
    {
        return false;
    }

    if (!EnsureInputShape(BatchSize))
    {
        return false;
    }

    const float MinAlphaRad = FMath::DegreesToRadians(MinAlphaDeg);
    const float MaxAlphaRad = FMath::DegreesToRadians(MaxAlphaDeg);
    const float A = FMath::Clamp(AlphaRad, MinAlphaRad, MaxAlphaRad);

    const float SinA = FMath::Sin(A);
    const float CosA = FMath::Cos(A);

    TArray<float> InputData;
    InputData.SetNumUninitialized(BatchSize * 5);

    for (int32 i = 0; i < BatchSize; ++i)
    {
        const FAeroSurfaceSampleV2& S = Samples[i];

        InputData[5 * i + 0] = S.Position.X;
        InputData[5 * i + 1] = S.Position.Y;
        InputData[5 * i + 2] = S.Side >= 0.0f ? 1.0f : -1.0f;
        InputData[5 * i + 3] = SinA;
        InputData[5 * i + 4] = CosA;
    }

    TArray<float> OutputData;
    OutputData.Init(0.0f, BatchSize);

    UE::NNE::FTensorBindingCPU InputBinding;
    InputBinding.Data = InputData.GetData();
    InputBinding.SizeInBytes = InputData.Num() * sizeof(float);

    UE::NNE::FTensorBindingCPU OutputBinding;
    OutputBinding.Data = OutputData.GetData();
    OutputBinding.SizeInBytes = OutputData.Num() * sizeof(float);

    TArray<UE::NNE::FTensorBindingCPU> Inputs;
    Inputs.Add(InputBinding);

    TArray<UE::NNE::FTensorBindingCPU> Outputs;
    Outputs.Add(OutputBinding);

    const auto Status = ModelInstance->RunSync(Inputs, Outputs);

    if (Status != UE::NNE::IModelInstanceRunSync::ERunSyncStatus::Ok)
    {
        UE_LOG(LogTemp, Warning, TEXT("AeroNNComponentV2: Cp inference failed"));
        return false;
    }

    OutCp = MoveTemp(OutputData);
    return true;
}

FAeroCoefficientsV2 UAeroNNComponentV2::IntegratePressureCoefficients(
    const TArray<float>& CpValues,
    float AlphaRad
) const
{
    FAeroCoefficientsV2 Coeffs;

    if (CpValues.Num() != SurfaceSamples.Num())
    {
        return Coeffs;
    }

    float Fx = 0.0f;
    float Fy = 0.0f;
    float Moment = 0.0f;

    const FVector2D RefPoint(MomentReferenceXPinn, MomentReferenceY);

    for (int32 i = 0; i < SurfaceSamples.Num(); ++i)
    {
        const float Cp = CpValues[i];
        const FAeroSurfaceSampleV2& S = SurfaceSamples[i];

        // Local pressure force coefficient on this surface segment.
        // dF* = -Cp * n * ds
        const float dFx = -Cp * S.Normal.X * S.DS;
        const float dFy = -Cp * S.Normal.Y * S.DS;

        Fx += dFx;
        Fy += dFy;

        // Moment coefficient around RefPoint.
        // 2D scalar cross product:
        // M = r_x * dF_y - r_y * dF_x
        const float rx = S.Position.X - RefPoint.X;
        const float ry = S.Position.Y - RefPoint.Y;

        Moment += rx * dFy - ry * dFx;
    }

    const float A = AlphaRad;

    const FVector2D DragDir(FMath::Cos(A), FMath::Sin(A));
    const FVector2D LiftDir(-FMath::Sin(A), FMath::Cos(A));

    Coeffs.FxCoeff = Fx;
    Coeffs.FyCoeff = Fy;
    Coeffs.CDPressure = Fx * DragDir.X + Fy * DragDir.Y;
    Coeffs.CL = Fx * LiftDir.X + Fy * LiftDir.Y;
    Coeffs.CM = Moment;

    // CD remains analytic for now.
    Coeffs.CD = ComputeAnalyticDrag(AlphaRad, Coeffs.CL, Coeffs.CDPressure);

    return Coeffs;
}

float UAeroNNComponentV2::ComputeAnalyticDrag(float AlphaRad, float CL, float CDPressure) const
{
    const float S = FMath::Sin(AlphaRad);

    // Pressure drag is kept as optional information, but not trusted alone.
    // Simple robust game drag:
    //   CD = CD0 + k CL^2 + post-stall-ish sin^2(alpha) term.
    return CD0 + InducedDragK * CL * CL + StallDragScale * S * S;
}

bool UAeroNNComponentV2::EvaluateAeroCoefficients(float AlphaRad, FAeroCoefficientsV2& Coeffs)
{
    Coeffs = FAeroCoefficientsV2();

    if (SurfaceSamples.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("AeroNNComponentV2: no surface samples"));
        return false;
    }

    const float MinAlphaRad = FMath::DegreesToRadians(MinAlphaDeg);
    const float MaxAlphaRad = FMath::DegreesToRadians(MaxAlphaDeg);
    const float AlphaEval = FMath::Clamp(AlphaRad, MinAlphaRad, MaxAlphaRad);

    TArray<float> CpValues;

    if (!EvaluateCpBatch(SurfaceSamples, AlphaEval, CpValues))
    {
        return false;
    }

    Coeffs = IntegratePressureCoefficients(CpValues, AlphaEval);
    return true;
}

bool UAeroNNComponentV2::EvaluateAero(float AlphaRad, float& CL, float& CD)
{
    CL = 0.0f;
    CD = 0.0f;

    FAeroCoefficientsV2 Coeffs;

    if (!EvaluateAeroCoefficients(AlphaRad, Coeffs))
    {
        return false;
    }

    CL = Coeffs.CL;
    CD = Coeffs.CD;

    return true;
}

bool UAeroNNComponentV2::EvaluateAeroWithMoment(float AlphaRad, float& CL, float& CD, float& CM)
{
    CL = 0.0f;
    CD = 0.0f;
    CM = 0.0f;

    FAeroCoefficientsV2 Coeffs;

    if (!EvaluateAeroCoefficients(AlphaRad, Coeffs))
    {
        return false;
    }

    CL = Coeffs.CL;
    CD = Coeffs.CD;
    CM = Coeffs.CM;

    return true;
}