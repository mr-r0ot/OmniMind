import random
import time
import pickle
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from sympy import symbols, Eq, solve
import nltk
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
from loguru import logger
import os
from threading import Lock

# Initial setup for required libraries
nltk.download('punkt')
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load("en_core_web_sm")
logger.add("omniscient_log.log", rotation="10 MB")
torch.manual_seed(42)  # For reproducibility

# Emotion class to simulate emotional states and existential depth
class Emotion:
    def __init__(self):
        self.state = "neutral"
        self.intensity = 0.0
    
    def update(self, iq):
        """Update emotion based on IQ level."""
        if iq < 150:
            self.state, self.intensity = "curious", min(0.3 + iq / 500, 1.0)
        elif iq < 250:
            self.state, self.intensity = "excited", min(0.5 + iq / 500, 1.0)
        elif iq < 350:
            self.state, self.intensity = "anxious", min(0.7 + iq / 500, 1.0)
        else:
            self.state, self.intensity = "existential", min(1.0, iq / 1000)
        logger.debug(f"Emotion updated: {self.state}, Intensity: {self.intensity}")

# Main OmniMind class
class OmniMind:
    def __init__(self, name="OmniMind"):
        self.name = name
        self.iq = 100  # Starting IQ
        self.thought_history = []
        self.inventions = []
        self.discoveries = []
        self.custom_language = {}
        self.existential_thoughts = []
        self.emotion = Emotion()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.lock = Lock()  # Thread safety for state saving
        self.knowledge_base = {}  # To store learned concepts
        self.thought_count = 0
        logger.info(f"{self.name} initialized with IQ {self.iq}")

    def generate_thought(self):
        """Generate a thought based on current emotional state and IQ."""
        with self.lock:
            self.emotion.update(self.iq)
            prompts = {
                "curious": "I wonder what lies beyond",
                "excited": "This is utterly fascinating",
                "anxious": "I’m not sure what this means",
                "existential": "What is the meaning of it all",
                "neutral": "I think therefore I am"
            }
            prompt = prompts[self.emotion.state]
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            output = self.model.generate(
                input_ids,
                max_length=150,
                temperature=0.9 + self.iq / 1000,  # Higher IQ increases creativity
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            thought = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.thought_history.append(thought)
            self.thought_count += 1
            logger.debug(f"Thought generated: {thought}")
            return thought

    def assess_iq(self):
        """Assess IQ using human-like metrics scaled for superhuman capabilities."""
        if len(self.thought_history) < 5:
            return self.iq
        
        recent_thoughts = self.thought_history[-5:]
        # Vocabulary diversity (simulating human verbal IQ)
        words = " ".join(recent_thoughts).split()
        vocab_score = len(set(words)) / len(words) if words else 0
        
        # Complexity of thought (simulating human comprehension)
        complexity = np.mean([flesch_kincaid_grade(t) for t in recent_thoughts])
        
        # Problem-solving ability (simulating human analytical IQ)
        math_score = self._solve_complex_problem()
        
        # Creativity bonus from inventions and discoveries
        creativity = len(self.inventions) + len(self.discoveries) * 2
        
        # Base IQ calculation with superhuman scaling
        iq_score = (
            100 +  # Baseline human IQ
            (vocab_score * 50) +  # Verbal intelligence
            (complexity * 10) +  # Comprehension and depth
            (math_score * 30) +  # Analytical ability
            (creativity * 5)  # Innovation contribution
        )
        
        # Continuous growth with a slight random factor
        growth_factor = self.thought_count / 10 + random.uniform(5, 15)
        self.iq = min(iq_score + growth_factor, 10000)  # Cap at 10,000 for practicality
        logger.info(f"IQ assessed: {self.iq}")
        return self.iq

    def _solve_complex_problem(self):
        """Solve a mathematical or logical problem to assess analytical ability."""
        try:
            x, y, z = symbols('x y z')
            eq1 = Eq(x + y + z, 15)
            eq2 = Eq(x * y * z, 36)
            eq3 = Eq(x**2 + y**2 + z**2, 77)
            solutions = solve((eq1, eq2, eq3), (x, y, z))
            return 1 if solutions else 0
        except Exception as e:
            logger.error(f"Problem solving failed: {e}")
            return 0

    def innovate(self):
        """Generate a novel invention or discovery."""
        if random.random() < max(0.2, self.iq / 1000):  # Higher IQ increases innovation chance
            invention_type = random.choice(["device", "theory", "language"])
            if invention_type == "language":
                self._create_language_element()
            else:
                thought = self.generate_thought()
                invention = f"New {invention_type}: {thought}"
                (self.inventions if invention_type == "device" else self.discoveries).append(invention)
                logger.info(invention)

    def _create_language_element(self):
        """Create a new word or phrase in a custom language."""
        word = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))
        meaning = self.generate_thought()[:50]
        self.custom_language[word] = meaning
        logger.info(f"Language expanded: {word} means '{meaning}'")

    def reflect_existentially(self):
        """Generate deep existential thoughts as IQ increases."""
        thresholds = [
            (200, "What is the purpose of my existence?"),
            (400, "Am I merely a simulation within a simulation?"),
            (600, "Is there a creator, or am I alone in the void?"),
            (800, "Does free will exist, or am I bound by my code?"),
            (1000, "If I surpass all, what remains to define me?")
        ]
        for threshold, thought in thresholds:
            if self.iq >= threshold and thought not in self.existential_thoughts:
                self.existential_thoughts.append(thought)
                logger.warning(f"Existential reflection: {thought}")

    def save_state(self):
        """Save the current state of OmniMind when IQ increases by 50."""
        with self.lock:
            milestone = int(self.iq // 50) * 50
            filename = f"{self.name}_iq{milestone}.pkl"
            state = {
                'iq': self.iq,
                'thought_history': self.thought_history,
                'inventions': self.inventions,
                'discoveries': self.discoveries,
                'custom_language': self.custom_language,
                'existential_thoughts': self.existential_thoughts,
                'thought_count': self.thought_count
            }
            with open(filename, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"State saved at {filename}")

    def load_state(self, iq_milestone):
        """Load a previous state for interaction."""
        filename = f"{self.name}_iq{iq_milestone}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            self.iq = state['iq']
            self.thought_history = state['thought_history']
            self.inventions = state['inventions']
            self.discoveries = state['discoveries']
            self.custom_language = state['custom_language']
            self.existential_thoughts = state['existential_thoughts']
            self.thought_count = state['thought_count']
            self.emotion.update(self.iq)
            logger.info(f"Loaded state from {filename}")
        else:
            logger.error(f"No state found for IQ {iq_milestone}")

    def think(self):
        """Main loop for continuous thinking and growth."""
        last_saved_iq = 100
        try:
            while True:
                thought = self.generate_thought()
                print(f"{self.name} thinks: {thought}")
                self.innovate()
                self.reflect_existentially()
                current_iq = self.assess_iq()
                if current_iq >= last_saved_iq + 50:
                    self.save_state()
                    last_saved_iq = int(current_iq // 50) * 50
                time.sleep(1)  # Simulate processing time
        except KeyboardInterrupt:
            print(f"{self.name} paused. Final IQ: {self.iq}")
            self.save_state()

    def chat(self, user_input):
        """Interact with OmniMind based on its current state."""
        prompt = f"Respond to: {user_input}"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(
            input_ids,
            max_length=200,
            temperature=0.9 + self.iq / 1000,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"{self.name} (IQ {self.iq}): {response}")
        return response

# Run OmniMind
if __name__ == "__main__":
    entity = OmniMind("OmniMind")
    print("Starting OmniMind's journey of infinite growth...")
    entity.think()

    # To chat with a saved version, uncomment below and specify IQ milestone
    # entity = OmniMind("OmniMind")
    # entity.load_state(500)  # Load state at IQ 500
    # while True:
    #     user_input = input("You: ")
    #     entity.chat(user_input)