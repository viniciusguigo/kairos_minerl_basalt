#!/bin/bash
# Generate videos for all tasks, all different agent conditions
evaluate_all_tasks () {
    # Evaluate FindCave
    export MINERL_GYM_ENV="MineRLBasaltFindCaveHighRes-v0"
    ./utility/evaluation_locally.sh --verbose

    # Evaluate MakeWaterfall
    export MINERL_GYM_ENV="MineRLBasaltMakeWaterfallHighRes-v0"
    ./utility/evaluation_locally.sh --verbose

    # Evaluate CreatePen
    export MINERL_GYM_ENV="MineRLBasaltCreateVillageAnimalPenHighRes-v0"
    ./utility/evaluation_locally.sh --verbose

    # Evaluate BuildHouse
    export MINERL_GYM_ENV="MineRLBasaltBuildVillageHouseHighRes-v0"
    ./utility/evaluation_locally.sh --verbose
}

# ------------------------------------------------------------------
# GENERAL SETUP
# ------------------------------------------------------------------
export MINERL_HEADLESS=1
export MINERL_MAX_EVALUATION_EPISODES=7
export EXPERIMENT_NAME='PAPER_EVALUATION'
export ENABLE_DEBUG_GUI='True'

# ------------------------------------------------------------------
# CONDITION: hybrid_navigation
# ------------------------------------------------------------------
export MODEL_OP_MODE='hybrid_navigation'
evaluate_all_tasks

# ------------------------------------------------------------------
# CONDITION: bc_only
# ------------------------------------------------------------------
export MODEL_OP_MODE='bc_only'
evaluate_all_tasks

# ------------------------------------------------------------------
# CONDITION: engineered_only
# ------------------------------------------------------------------
export MODEL_OP_MODE='engineered_only'
evaluate_all_tasks
