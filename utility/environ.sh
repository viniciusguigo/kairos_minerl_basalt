#!/bin/bash
export MINERL_DATA_ROOT="data/"
export IMAGE_NAME="aicrowd/neurips2021-minerl-challenge"
export IMAGE_TAG="agent"
export MINERL_HEADLESS=1
export MINERL_MAX_EVALUATION_EPISODES=2

### AVAILABLE TASKS
# MineRLBasaltFindCaveHighRes-v0
# MineRLBasaltMakeWaterfallHighRes-v0
# MineRLBasaltCreateVillageAnimalPenHighRes-v0
# MineRLBasaltBuildVillageHouseHighRes-v0
export MINERL_GYM_ENV="MineRLBasaltMakeWaterfallHighRes-v0"