from HRT.agents import QLearningAgent, Parameters
from HRT.field import field, field2


def main():
    start_position = (1, 1)
    parameters = Parameters(
        action_choice_strategy=Parameters.ActionChoiceStrategy.GREEDY,
        state_expression=Parameters.StateExpression.COORDINATE,
        max_trial=100,
        max_step=10000,
        show_step_details=True,
        reward=10.0,
        q_init_value=0.0
    )
    agent = QLearningAgent(field, start_position, parameters)

    print("The best step counts of all trials:", agent.run())


if __name__ == "__main__":
    main()
