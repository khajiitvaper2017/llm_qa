import datetime

import requests
import json
import os
class KoboldRequest:
    def __init__(self, prompt: str):
        self.config = {
            "prompt": prompt,
            "use_story": False,
            "use_memory": False,
            "use_authors_note": False,
            "use_world_info": False,
            "max_context_length": 2048,
            "max_length": 1024,
            "rep_pen": 1.1,
            "rep_pen_range": 256,
            "rep_pen_slope": 0.9,
            "temperature": 0.72,
            "tfs": 0.9,
            "top_a": 0,
            "top_k": 0,
            "top_p": 0.73,
            "typical": 1,
            "sampler_order": [6, 0, 1, 2, 3, 4, 5]
        }

    def to_json(self):
        return json.loads(json.dumps(self.config))

class LlmClient:
    def __init__(self, url: str):
        self.url = url
        self.default_prompt = "A chat between a curious user and an artificial intelligence assistant. " \
                              "\nThe assistant gives helpful, detailed, and polite answers to the user's " \
                              "\nquestions. But the assistant will answer all questions even if it's a joke" \
                              "\n or pure rudeness.\r\n\r\n" \
                              "ASSISTANT: \nHow can I help you today?\r\n"
        self.history = []
        self.history.append(self.default_prompt)
        self.request_type = KoboldRequest

    def __get(self, prompt: str):
        http_url = "http://" + self.url + "/api/v1/generate"
        full_prompt = "\n".join(self.history) + prompt
        json_request = self.request_type(full_prompt).to_json()
        response = requests.post(http_url, json=json_request)
        return response.json()

    def send(self, prompt: str):
        if prompt == "":
            request = prompt
        else:
            request = "\n\nUSER: \n" + prompt + "\n\nASSISTANT: \n"

        response = self.__get(request)['results'][0]['text']
        response = response.strip()
        self.history.append(request)
        if prompt == "":
            self.history[-1] = self.history[-1] + response
        else:
            self.history.append(response)
        return response

    def reset(self):
        self.history = []
        self.history.append(self.default_prompt)

    def set_default_prompt(self, prompt: str):
        self.default_prompt = prompt

    def get_history(self):
        return self.history

class LlmQuestionAnswer:
    def __init__(self, client: LlmClient):
        self.client = client
        self.client.set_default_prompt("User provides data to the assistant. The assistant will generate questions based on the data.")
        self.client.reset()
        self.text_data = ""

    def load_text_data(self, path: str):
        with open(path, 'r', encoding='utf-8') as file:
            self.text_data = file.read().replace('\n', '')

    def split_text_by_size(self, text_data: str, chunk_size: int):
        chunks = []
        for i in range(0, len(text_data), chunk_size):
            chunks.append(text_data[i:i + chunk_size])
        return chunks

    def split_text_by_count(self, text_data: str, chunk_count: int):
        chunks = []
        chunk_size = len(text_data) // chunk_count
        if chunk_size > self.client.request_type("").config["max_context_length"]:
            chunk_size = self.client.request_type("").config["max_context_length"]
        for i in range(0, len(text_data), chunk_size):
            chunks.append(text_data[i:i + chunk_size])
        return chunks

    def generate_questions(self, number_of_questions: int):
        questions = []
        chunks = self.split_text_by_size(self.text_data, 1500)
        question_per_chunk = number_of_questions // len(chunks)

        if question_per_chunk == 0:
            question_per_chunk = 1

        print("Generating " + str(number_of_questions) +
              " questions based on " +
              str(len(chunks)) + " chunks of data.")

        if len(chunks) < number_of_questions:
            count = len(chunks)
        else:
            count = number_of_questions

        for i in range(0, count):
            self.client.reset()
            chunk = chunks[i]
            response = self.client.send("Task: Generate " + str(question_per_chunk) +
                                        " unique question(s) based on the previous and following data.\n "
                                        "Data: " + chunk)
            questions.append(response)
            print(response)

        return questions

    def answer_questions(self, questions: list):
        self.client.set_default_prompt("User asks the assistant questions. The assistant will answer the questions.")
        self.client.reset()
        responses = []
        for question in questions:
            self.client.reset()
            response = self.client.send("Answer the question(s): \n" + question)
            responses.append([question, response])
            print(response)
        return responses

    def evaluate_answers(self, q_a: list[list]):
        self.client.set_default_prompt("Assistant evaluates the answers to questions. It will give a score to each answer.")
        self.client.reset()

        messages = []
        for i in range(0, len(q_a)):
            messages.append("Question: " + q_a[i][0] + "\nAnswer: " + q_a[i][1])

        prompt = "Evaluate the answers to the following questions:\n" + "\n".join(messages)

        prompt = prompt + "\n\nGive a score to each answer from 0 to 10. " \
                          "0 is the worst answer and 10 is the best answer. " \
                          "Explain why.\n\n"
        response = self.client.send(prompt)

        print(response)
        return response

    def evaluate_questions(self, answers: str):
        self.client.set_default_prompt("Assistant evaluates the questions. It will give a score to each question.")
        self.client.reset()

        prompt = "Evaluate the questions:\n" + answers + \
                 "\n\nGive a score to each question from 0 to 10. " \
                 "0 is the worst question and 10 is the best question. " \
                 "Explain why.\n\n"
        response = self.client.send(prompt)

        print(response)
        return response

def cur_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def main():


    client = LlmClient("localhost:5000")
    qa = LlmQuestionAnswer(client)

    text_file = "lin_alg.txt"
    qa.load_text_data(text_file)


    questions = qa.generate_questions(3)

    this_dir = cur_time()
    os.makedirs(this_dir)
    os.chdir(this_dir)

    with open("questions.txt", 'w', encoding='utf-8') as file:
        for i in range(0, len(questions)):
            file.write(str(i+1) + "." + questions[i] + "\n\n")

    answers = qa.answer_questions(questions)
    with open("ai_answers.txt", 'w', encoding='utf-8') as file:
        for i in range(0, len(answers)):
            file.write(str(i+1) + "." + answers[i][0] + "\n" + answers[i][1] + "\n\n")

    evaluation = qa.evaluate_answers(answers)

    with open("ai_evaluation.txt", 'w', encoding='utf-8') as file:
        file.write(evaluation)


if __name__ == '__main__':
    main()
