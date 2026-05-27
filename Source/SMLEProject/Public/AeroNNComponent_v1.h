#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "NNEModelData.h"
#include "NNERuntimeRunSync.h"
#include "AeroNNComponent_v1.generated.h"


UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SMLEPROJECT_API UAeroNNComponent_v1 : public UActorComponent
{
    GENERATED_BODY()

public:
    UAeroNNComponent_v1();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero NN")
    UNNEModelData* ModelData = nullptr;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero NN")
    float AlphaScale = 1.3962634f; // 80 deg in radians

    UFUNCTION(BlueprintCallable, Category = "Aero NN")
    bool EvaluateAero(float AlphaRad, float& CL, float& CD);

protected:
    virtual void BeginPlay() override;

private:
    bool bReady = false;

    TSharedPtr<UE::NNE::IModelInstanceRunSync> ModelInstance;
};