import nltk
import time
import speech_recognition as sr

# download the required data files using the NLTK downloader
nltk.download("stopwords")  # library of words to ignore such as "the", "a", "an", "in", etc.
nltk.download("vader_lexicon")  # nlp processing library

# set up the speech recognition library to listen to the microphone input
r = sr.Recognizer()
mic = sr.Microphone()

class PushdownAutomaton:
    """
    A class that represents a pushdown automaton (PDA)
    """

    def __init__(self, initial_state, transitions, stack_symbols, final_states=None):
        self.initial_state = initial_state
        self.transitions = transitions
        self.stack_symbols = stack_symbols
        self.final_states = final_states or []

    def process_input(self, input_symbols, verbose=False):
        # initialize the current state and stacks
        current_state = self.initial_state
        stacks = [[] for _ in range(len(self.stack_symbols))]

        # iterate over the input symbols
        for symbol in input_symbols:
            # look up the transition for the current state and input symbol
            transition = self.transitions.get((current_state, symbol))

            # if a transition exists, update the current state and stack
            if transition:
                current_state, stack_operations = transition
                for i, (operation) in enumerate(stack_operations):
                    if operation == "push":
                        stacks[i].append(symbol)
                    elif operation == "pop":
                        stacks[i].pop()
                    if verbose:
                        print(f"State: {current_state}, Stack: {stacks}")
        return current_state

    def is_final(self):
        return self.current_state in self.final_states


# define the PDA
pda = PushdownAutomaton(
    initial_state=0,
    transitions={
        # (current_state, input_word): (next_state, (stack_operation, stack_symbol))

        (0, "happy"): (1, [("push", "p"), ("push", "e")]),
        (0, "sad"): (2, [("push", "n")]),
        (0, "neutral"): (3, [("push", "n"), ("pop", "p"), ("pop", "e")]),
        (0, "love"): (1, [("push", "p"), ("push", "l")]),
        (0, "hate"): (2, [("push", "n"), ("push", "h")]),
        (0, "angry"): (2, [("push", "n"), ("push", "a")]),
        (0, "excited"): (4, [("push", "e")]),
        (0, "dislike"): (2, [("push", "n"), ("push", "d")]),
        (1, "happy"): (1, [("push", "p"), ("push", "e")]),
        (2, "sad"): (2, [("push", "n")]),
        (3, "neutral"): (3, [("push", "n"), ("pop", "p"), ("pop", "e")]),
        (4, "excited"): (4, [("push", "e")]),
    },
    stack_symbols=["p", "n", "e", "l", "h", "a", "d"],
    final_states=[1, 2, 3, 4]
)


# define a function that takes an input sentence and returns the emotion reflected in the sentence
# using the PDA
def determine_emotion(sentence):
    # use NLTK to tokenize the input sentence into a list of words
    words = nltk.word_tokenize(sentence)
    words = [word for word in words if word not in nltk.corpus.stopwords.words("english")]

    # keep track of the number of times each emotion occurs in the sentence
    emotion_counts = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "excited": 0,
    }

    # process each word in the input sentence separately
    for word in words:
        # use the PDA to process the input word
        current_state = pda.process_input([word])

        # check the current state of the PDA and increment the corresponding emotion count
        if current_state == 1:
            emotion_counts["positive"] += 1
        elif current_state == 2:
            emotion_counts["negative"] += 1
        elif current_state == 3:
            emotion_counts["neutral"] += 1
        elif current_state == 4:
            emotion_counts["excited"] += 1

    # return the emotion that occurs most frequently in the sentence
    if sum(emotion_counts.values()) > 0:
        return max(emotion_counts, key=emotion_counts.get)
    else:
        # otherwise, return "cannot determine"
        return "cannot determine"


# ----------- MAIN --------------
# sleep for 1 second to prevent mixing output with nltk download statements
time.sleep(1)

while True:
    # listen to microphone input
    with mic as source:
        print("Listening... ")
        audio = r.listen(source)
    statement = r.recognize_google(audio)

    print(f"Statement: {statement}")
    print(f"Emotion detected: {determine_emotion(statement)}")