#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "NNEModelData.h"
#include "NNERuntimeRunSync.h"
#include "AeroNNComponentV2.generated.h"

USTRUCT(BlueprintType)
struct FAeroSurfaceSampleV2
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
    FVector2D Position = FVector2D::ZeroVector; // x_pinn, y, chord-normalized

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
    FVector2D Normal = FVector2D::ZeroVector; // outward normal

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
    float DS = 0.0f; // normalized surface length element

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
    float Side = 1.0f; // +1 upper, -1 lower
};

USTRUCT(BlueprintType)
struct FAeroCoefficientsV2
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Aero")
    float CL = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Aero")
    float CD = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Aero")
    float CDPressure = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Aero")
    float FxCoeff = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Aero")
    float FyCoeff = 0.0f;
};

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SMLEPROJECT_API UAeroNNComponentV2 : public UActorComponent
{
    GENERATED_BODY()

public:
    UAeroNNComponentV2();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero NN")
    UNNEModelData* ModelData = nullptr;

    // Number of points per side before midpoint integration.
    // 900 matches the current Python diagnostic geometry.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero Geometry")
    int32 NacaPointCount = 900;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero Geometry")
    float NacaM = 0.02f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero Geometry")
    float NacaP = 0.4f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero Geometry")
    float NacaT = 0.12f;

    // Clamp because the NN should not extrapolate too far outside training data.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero Runtime")
    float MinAlphaDeg = -25.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero Runtime")
    float MaxAlphaDeg = 25.0f;

    // Simple analytic drag until we train / import a better drag model.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero Drag")
    float CD0 = 0.02f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero Drag")
    float InducedDragK = 0.05f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero Drag")
    float StallDragScale = 0.8f;

    UFUNCTION(BlueprintCallable, Category = "Aero NN")
    bool EvaluateAero(float AlphaRad, float& CL, float& CD);

    UFUNCTION(BlueprintCallable, Category = "Aero NN")
    bool EvaluateAeroCoefficients(float AlphaRad, FAeroCoefficientsV2& Coeffs);

    UFUNCTION(BlueprintCallable, Category = "Aero NN")
    bool EvaluateCpAtPoint(float XPinn, float Y, float Side, float AlphaRad, float& Cp);

protected:
    virtual void BeginPlay() override;

private:
    bool InitModel();
    bool EnsureInputShape(int32 BatchSize);
    void BuildNACASurface();

    bool EvaluateCpBatch(const TArray<FAeroSurfaceSampleV2>& Samples, float AlphaRad, TArray<float>& OutCp);
    FAeroCoefficientsV2 IntegratePressureCoefficients(const TArray<float>& CpValues, float AlphaRad) const;

    float ComputeAnalyticDrag(float AlphaRad, float CL, float CDPressure) const;

    static float NacaCamberY(float X01, float M, float P);
    static void NacaUpperLower(float X01, float M, float P, float T, float& Yu, float& Yl);

private:
    bool bReady = false;

    int32 CurrentBatchSize = INDEX_NONE;

    TSharedPtr<UE::NNE::IModelInstanceRunSync> ModelInstance;

    TArray<FAeroSurfaceSampleV2> SurfaceSamples;
};