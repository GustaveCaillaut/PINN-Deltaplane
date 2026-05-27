#include "AeroNNComponent_v1.h"

#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNEModelData.h"

UAeroNNComponent_v1::UAeroNNComponent_v1()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UAeroNNComponent_v1::BeginPlay()
{
    Super::BeginPlay();

    if (!ModelData)
    {
        UE_LOG(LogTemp, Error, TEXT("AeroNNComponent: ModelData is null"));
        return;
    }

    TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));

    if (!Runtime.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("AeroNNComponent: Could not get NNERuntimeORTCpu"));
        return;
    }

    TSharedPtr<UE::NNE::IModelCPU> Model = Runtime->CreateModelCPU(ModelData);

    if (!Model.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("AeroNNComponent: Could not create model"));
        return;
    }

    ModelInstance = Model->CreateModelInstanceCPU();

    if (!ModelInstance.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("AeroNNComponent: Could not create model instance"));
        return;
    }

    TArray<UE::NNE::FTensorShape> InputShapes;
    InputShapes.Add(UE::NNE::FTensorShape::Make({ 1, 1 }));

    if (ModelInstance->SetInputTensorShapes(InputShapes) != UE::NNE::IModelInstanceRunSync::ESetInputTensorShapesStatus::Ok)
    {
        UE_LOG(LogTemp, Error, TEXT("AeroNNComponent: Failed to set input shape"));
        return;
    }

    bReady = true;
    UE_LOG(LogTemp, Display, TEXT("AeroNNComponent: ready"));
}

bool UAeroNNComponent_v1::EvaluateAero(float AlphaRad, float& CL, float& CD)
{
    CL = 0.0f;
    CD = 0.0f;

    if (!bReady || !ModelInstance.IsValid())
    {
        return false;
    }

    float InputData[1] = { AlphaRad / AlphaScale };
    float OutputData[2] = { 0.0f, 0.0f };

    UE::NNE::FTensorBindingCPU InputBinding;
    InputBinding.Data = InputData;
    InputBinding.SizeInBytes = sizeof(InputData);

    UE::NNE::FTensorBindingCPU OutputBinding;
    OutputBinding.Data = OutputData;
    OutputBinding.SizeInBytes = sizeof(OutputData);

    TArray<UE::NNE::FTensorBindingCPU> Inputs;
    Inputs.Add(InputBinding);

    TArray<UE::NNE::FTensorBindingCPU> Outputs;
    Outputs.Add(OutputBinding);

    auto Status = ModelInstance->RunSync(Inputs, Outputs);

    if (Status != UE::NNE::IModelInstanceRunSync::ERunSyncStatus::Ok)
    {
        UE_LOG(LogTemp, Warning, TEXT("AeroNNComponent: inference failed"));
        return false;
    }

    CL = OutputData[0];
    CD = OutputData[1];

    return true;
}