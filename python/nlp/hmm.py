# Part Of Speach Tagging with Hidden Markov Models
# use train to train model an save in zip file
# use classify to apply model to a sentence

from __future__ import annotations  # Python 3.10 style annotations

import math
import sys
from tqdm import tqdm
from collections import defaultdict
import itertools
import json
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    if sys.argv[1] == "train":
        if len(sys.argv) > 3:
            train(file_path=sys.argv[2], model_file_path=sys.argv[3])
        else:
            train(file_path="BIO_formals_blank.txt", model_file_path="model4.zip")
    elif sys.argv[1] == "classify":
        if len(sys.argv) > 3:
            classify(model_file_path=sys.argv[2], text=sys.argv[3])
        else:
            classify(model_file_path="model4.zip", text="The sound was music to her ears .")
    else:
        print(sys.argv[1], "not available")


def train(file_path: str, model_file_path: str):

    word_tag_counter = defaultdict(int)
    start_tag_counter = defaultdict(int)
    state_transition_counter = defaultdict(int)
    state_emission_counter = defaultdict(lambda: defaultdict(int))
    prev_tag = "<$>"
    tags = set()

    next_token_is_start_of_sentence = True
    with open(file_path, mode='r', encoding="utf-8") as file:
        text = file.readlines()

    for line in tqdm(text):
        line: str
        if line.isspace():
            next_token_is_start_of_sentence = True
            prev_tag = "<$>"
        else:
            (word, tag) = line.strip().split(' ')
            word_tag_counter[(word, tag)] += 1
            state_emission_counter[tag][word] += 1
            tags.add(tag)
            state_transition_counter[(prev_tag, tag)] += 1
            prev_tag = tag
            if next_token_is_start_of_sentence:  # here it is not the next token, but the current one
                start_tag_counter[tag] += 1
            next_token_is_start_of_sentence = False

    a_priori = a_priori_prob(start_tag_counter, tags)
    transition_prob = state_transition_prob(state_transition_counter, tags)
    emission_prob = state_emission_prob(state_emission_counter)
    result = {
        "a_priori": a_priori,
        "state_transition": transition_prob,
        "state_emission": emission_prob
    }
    print_all_probs(result)
    temp_file_num, temp_path = tempfile.mkstemp()
    with open(temp_path, mode="w") as temp_file:
        json.dump(result, indent=1, fp=temp_file)
    with ZipFile(model_file_path, mode='w', compression=ZIP_DEFLATED) as zipfile:
        zipfile.write(temp_path)  # storing model in compressed form
    # j = json.dumps(result, indent=1)  # , fp=open(model_file_path, mode='x', encoding="utf-8"))


def print_all_probs(result: dict[str, dict]):
    pass
    # for level1_key, level1_value in result.items():
    #     print(level1_key)
    #     for level2_item in level1_value.items():
    #         print(level2_item)
    #     print("-" * 24)


def a_priori_prob(start_tag_counter: defaultdict, tags: set) -> dict:
    """
    calculate a-priori probabilities for tags
    """

    start_tag_counter = start_tag_counter.copy()
    for tag in tags:
        start_tag_counter[tag] += 1  # add-one-smoothing
    total = sum(start_tag_counter.values())
    for tag in tags:
        start_tag_counter[tag] /= total
    result = dict(start_tag_counter)

    return result


def state_transition_prob(state_transition_counter: defaultdict[(str, str), int], tags: set[str]) \
        -> dict[str, dict[str, float]]:
    """
    calculate probabilities for transition from one tag to another tag
    """
    prob_per_current_tag_per_prev_tag: dict[str, dict[str, int | float]] = defaultdict(lambda: defaultdict(int))

    for ((prev_tag, tag), value) in state_transition_counter.items():
        prob_per_current_tag_per_prev_tag[prev_tag][tag] = value

    for _, probs_for_this_prev_tag in prob_per_current_tag_per_prev_tag.items():
        for current_tag in tags:
            probs_for_this_prev_tag[current_tag] += 1
        total = sum(probs_for_this_prev_tag.values())
        for current_tag in tags:
            probs_for_this_prev_tag[current_tag] /= total

    return prob_per_current_tag_per_prev_tag


def state_emission_prob(state_emission_counter: defaultdict[defaultdict[str, int | float]]) -> dict:
    """
    calculate probabilities for emission per tag
    """
    known_words = set()
    for x in state_emission_counter.keys():
        known_words |= state_emission_counter[x].keys()

    prob_per_emission_per_tag = {}
    for tag, count_per_tag in state_emission_counter.items():
        total = sum(count_per_tag.values())
        prob_per_emission_per_tag[tag] = count_per_tag.copy()
        for emission in prob_per_emission_per_tag[tag].keys():
            prob_per_emission_per_tag[tag][emission] = (prob_per_emission_per_tag[tag][emission] + 1) / (
                        total + len(known_words))
        rest: set = known_words - prob_per_emission_per_tag[tag].keys()
        for emission in rest:
            prob_per_emission_per_tag[tag][emission] = 1 / (
                    total + len(known_words))
    return prob_per_emission_per_tag


def classify(model_file_path: str, text: str):
    # model entgegen nehmen
    with ZipFile(model_file_path, mode="r") as zip_file:
        with zip_file.open(zip_file.namelist()[0]) as model_file:
            model = json.load(model_file)
    a_priori: dict[str, float] = model["a_priori"]
    state_transition: dict[str, dict[str, float]] = model["state_transition"]
    state_emission: dict[str, dict[str, float]] = model["state_emission"]

    viterbi_chart: dict[int, dict[str, (float, str)]] = defaultdict(dict)  # timestep -> tag -> (score, prev_tag)

    words = text.split(' ')
    known_words = set()
    for x in state_emission.values():
        known_words |= x.keys()
    # Schritt 0 mit apriori
    for tag, a_priori_probability in a_priori.items():
        viterbi_chart[0][tag] = \
            (
                math.log(a_priori_probability)
                + emission_score(state_emission, tag=tag, word=words[0], known_words=known_words),
                "<$>"
            )

    for time_step, word in itertools.islice(enumerate(words), 1, None):
        valid_states = set()
        for potential_tag in viterbi_chart[time_step - 1].keys():
            # all states that were reachable in the last time step
            valid_states.add(potential_tag)

        reachable_states = set()
        for valid_prev_state in valid_states:
            for potential_state, prob in state_transition[valid_prev_state].items():
                if prob != 0:  # which states can we reach from the valid states
                    reachable_states.add(potential_state)

        for reachable_state in reachable_states:
            max_score, prev_state = viterbi_update(reachable_state,
                                                   time_step,
                                                   word,
                                                   viterbi_chart,
                                                   state_transition,
                                                   state_emission,
                                                   known_words
                                                   )
            viterbi_chart[time_step][reachable_state] = (max_score, prev_state)
    tag_sequence, score = follow_backpointer(viterbi_chart)
    print(' '.join(tag_sequence))
    print(score)


def viterbi_update(state: str,
                   time_step,
                   observation: str,
                   chart: dict[int, dict[str, (float, str)]],
                   state_transition: dict[str, dict[str, float]],
                   state_emission: dict[str, dict[str, float]],
                   known_words: set[str],
                   ) -> [float, str]:
    """
    calculate scores along the path with best incoming edges
    :param known_words: needed for node score
    :param state_emission:
    :param state_transition: contains probabilities, not scores
    :param state: current considered state
    :param time_step: current step
    :param observation: the word
    :param chart: viterbi chart
    :return: score, state
    """
    # 1. Access  the scores for all possible incoming states (you get these from the transducer)
    # from the chart (access it at the given timeStep)
    possible_incoming_states = chart[time_step - 1].keys()

    # 2. Add the edge scores for a transition from the previous states to state
    # • Yet again, the transducer has that information stored
    best_incoming_state = None
    best_score = -math.inf

    # 3. Find the previous state, which now have the highest score
    for incoming_state in possible_incoming_states:
        score = math.log(state_transition[incoming_state][state]) + chart[time_step - 1][incoming_state][0]
        # [0] for the score
        if score > best_score:
            best_score = score
            best_incoming_state = incoming_state

    # 4. Add the node score (ask the Transducer) to the score
    result_score = best_score + emission_score(state_emission, word=observation, tag=state, known_words=known_words)

    # 5. Return the tuple consisting of (score,state) for the maximum
    return result_score, best_incoming_state


def follow_backpointer(viterbi_chart: dict[int, dict[str, (float, str)]]) -> (list[str], float):
    """
    choose best score and sequence of tags
    """
    indices: list[int] = sorted(viterbi_chart.keys())
    last_index = indices[-1]
    best_state = None
    prev_of_best_state = None
    best_score = -math.inf
    for state, (score, prev) in viterbi_chart[last_index].items():
        if score > best_score:
            best_score = score
            best_state = state
            prev_of_best_state = prev

    result = [best_state, prev_of_best_state]  # reverse later
    prev_state = prev_of_best_state
    for index in itertools.islice(reversed(indices), 1, None):
        _, prev_state = viterbi_chart[index][prev_state]
        if prev_state != "<$>":
            result.append(prev_state)

    return list(reversed(result)), best_score


def emission_score(state_emission: dict[str, dict[str, float]], word: str, tag: str, known_words: set[str]) -> float:
    """
    "Also at runtime, unknown emissions may appear in the input that were not seen during
    training. Calculate an emission probability of 1 / (1 + number_of_emissions_known_to_the_model) for these."
    """

    if word in known_words:
        if state_emission[tag].get(word):
            return math.log(state_emission[tag].get(word))
        else:
            return math.log(1.0 / (1.0 + len(known_words)))  # sollte nie aufgerufen werden mit unserem Training

    else:
        return math.log(1.0 / (1.0 + len(known_words)))


if __name__ == "__main__":
    main()
