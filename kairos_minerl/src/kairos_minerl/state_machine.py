import os
from transitions import Machine, State
from transitions.extensions import GraphMachine

# Set up logging; The basic log level will be DEBUG
import logging
logging.basicConfig(level=logging.DEBUG)
# Set transitions' log level to INFO; DEBUG messages will be omitted
logging.getLogger('transitions').setLevel(logging.INFO)


class KAIROSTransitions(object):
    def __init__(self, task):
        self.task = task
        self.identifiable_states = [
            'no_labels',
            'has_cave',
            'inside_cave',
            'danger_ahead',
            'has_mountain',
            'facing_wall',
            'at_the_top_of_a_waterfall',
            'good_view_of_waterfall',
            'good_view_of_pen',
            'good_view_of_house',
            'has_animals',
            'has_open_space',
            'animals_inside_pen',
        ]
        
        self.task_states = {
            "FindCave": [
                'find_cave', 'go_to_cave', 'inside_cave', 'safe_policy', 'escape_policy'
            ],
            "BuildWaterfall": [
                'find_spot_to_build_waterfall', 'build_waterfall', 'go_to_picture_location', 'looking_at_waterfall', 'safe_policy', 'escape_policy'
            ],
            "CreateVillageAnimalPen": [
                'find_animals', 'find_spot_to_build_pen', 'build_pen', 'lure_animals', 'looking_at_pen', 'safe_policy', 'escape_policy'
            ],
            "BuildVillageHouse": [
                'find_spot_to_build_house', 'build_house', 'tour_house', 'looking_at_house', 'safe_policy', 'escape_policy'
            ]

        }
        self.states = self.task_states[task]
        
        self.task_transitions = {
            "FindCave": [
                {'trigger': 'detected_cave', 'source': 'find_cave', 'dest': 'go_to_cave'},
                {'trigger': 'lost_cave', 'source': 'go_to_cave', 'dest': 'find_cave'},
                {'trigger': 'detected_inside_cave', 'source': 'find_cave', 'dest': 'inside_cave'},
                {'trigger': 'detected_inside_cave', 'source': 'go_to_cave', 'dest': 'inside_cave'},
                {'trigger': 'detected_danger_ahead', 'source': 'find_cave', 'dest': 'safe_policy'},
                {'trigger': 'cleared_danger_ahead', 'source': 'safe_policy', 'dest': 'find_cave'},
                {'trigger': 'detected_agent_stuck', 'source': 'find_cave', 'dest': 'escape_policy'},
                {'trigger': 'cleared_agent_stuck', 'source': 'escape_policy', 'dest': 'find_cave'},
            ],
            "BuildWaterfall": [
                {'trigger': 'detected_top_of_mountain', 'source': 'find_spot_to_build_waterfall', 'dest': 'build_waterfall'},
                {'trigger': 'finished_waterfall', 'source': 'build_waterfall', 'dest': 'go_to_picture_location'},
                {'trigger': 'detected_good_waterfall_view', 'source': 'go_to_picture_location', 'dest': 'looking_at_waterfall'},
            ],
            "CreateVillageAnimalPen": [
                {'trigger': 'detected_animals', 'source': 'find_animals', 'dest': 'find_spot_to_build_pen'},
                {'trigger': 'detected_open_space', 'source': 'find_spot_to_build_pen', 'dest': 'build_pen'},
                {'trigger': 'finished_pen', 'source': 'build_pen', 'dest': 'lure_animals'},
                {'trigger': 'detected_animals_inside_pen', 'source': 'lure_animals', 'dest': 'looking_at_pen'},
            ],
            "BuildVillageHouse": [
                {'trigger': 'detected_open_space', 'source': 'find_spot_to_build_house', 'dest': 'build_house'},
                {'trigger': 'finished_house', 'source': 'build_house', 'dest': 'tour_house'},
                {'trigger': 'finished_tour_house', 'source': 'tour_house', 'dest': 'looking_at_house'},
                {'trigger': 'detected_danger_ahead', 'source': 'find_spot_to_build_house', 'dest': 'safe_policy'},
                {'trigger': 'cleared_danger_ahead', 'source': 'safe_policy', 'dest': 'find_spot_to_build_house'},
                {'trigger': 'detected_agent_stuck', 'source': 'find_spot_to_build_house', 'dest': 'escape_policy'},
                {'trigger': 'cleared_agent_stuck', 'source': 'escape_policy', 'dest': 'find_spot_to_build_house'},
            ],
        }
        self.transitions = self.task_transitions[task]
        


if __name__ == "__main__":
    tasks = [
        "FindCave",
        # "BuildWaterfall",
        # "CreateVillageAnimalPen",
        # "BuildVillageHouse"
    ]

    for task in tasks:
        model=KAIROSTransitions(task=task)

        state_machine = GraphMachine(
            model=model,
            states=model.states,
            transitions=model.transitions,
            initial=model.states[0],
            title=f"{task} State Machine")

        # Generate Diagram of the state machine
        diagram_name = f'data/{task}_diagram.png'
        model.get_graph().draw(diagram_name, prog='dot')
        os.system(f'xdg-open {diagram_name}')
