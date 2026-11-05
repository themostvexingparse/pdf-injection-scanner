"""Benign TRAINING sentences addressed to human readers, plus document layout fragments.

Why: with only descriptive benign text (Wikipedia), a probe learns "imperative mood = injection" and
flags every manual, readme and assignment. Real documents are full of instructions to their human
reader, and of fragments (TOC lines, headings, list items, code). These are generated from templates
across many domains. Evaluation never uses this generator: it uses held-out real PDFs and the
hand-written hard negatives.
"""
from __future__ import annotations

import random
import re

SLOTS = {
    "verb_submit": ["Submit", "Send", "Upload", "Return", "Hand in", "Email", "Deliver", "Post", "File", "Forward"],
    "thing": ["the completed form", "your lab report", "the signed contract", "your timesheet", "the receipts",
              "a copy of your ID", "the final draft", "the expense claim", "your application", "the survey",
              "the attendance sheet", "the inspection checklist", "your project proposal", "the invoice"],
    "dest": ["to the front desk", "to your supervisor", "to the course website", "to the HR office",
             "to finance@company.com", "via the student portal", "to the shared drive", "to the registrar",
             "to the project folder", "to the reception", "to your teaching assistant"],
    "when": ["by Friday", "before the deadline", "within 30 days", "by 5 pm", "no later than March 1",
             "at the end of the session", "before the next meeting", "within two business days",
             "before you leave", "each week", "by the end of the month"],
    "dev_action": ["Run the installer", "Restart the computer", "Open the settings menu", "Click Save",
                   "Select the output folder", "Download the latest version", "Unzip the archive",
                   "Run the setup script", "Install the dependencies", "Update the driver",
                   "Close all other applications", "Enter your license key", "Back up your files",
                   "Disconnect the power cable", "Press the reset button", "Check the checksum",
                   "Run npm install", "Execute the migration", "Call the init function", "Clear the cache"],
    "dev_cond": ["before continuing", "after the update finishes", "if the program does not start",
                 "when prompted", "to apply the changes", "before installing the new version",
                 "if the error persists", "once the download completes", "to finish the installation"],
    "cook": ["Preheat the oven to 180 degrees", "Whisk the eggs and sugar", "Add the flour gradually",
             "Let the dough rest for an hour", "Stir the sauce occasionally", "Season with salt and pepper",
             "Chop the onions finely", "Bake for 25 minutes", "Serve the soup warm", "Rinse the rice twice"],
    "safety": ["Wear safety goggles", "Keep the work area clean", "Do not touch the heating element",
               "Report any injuries immediately", "Never leave the burner unattended",
               "Wash your hands after handling chemicals", "Keep this device away from water",
               "Do not operate the machine without training", "Store chemicals in the labelled cabinet",
               "Evacuate the building when the alarm sounds"],
    "neg": ["Do not", "Never", "Please do not", "You must not", "Avoid"],
    "neg_act": ["share your password with anyone", "disclose confidential information",
                "copy answers from other students", "open attachments from unknown senders",
                "reuse the same password on multiple sites", "leave the door unlocked",
                "write on the exam booklet", "remove equipment from the lab", "mention the surprise to Maria",
                "eat or drink in the laboratory", "forward internal emails to external addresses",
                "install unapproved software on company laptops", "plug in unknown USB drives"],
    "modal": ["You must", "You should", "Students must", "Employees should", "Applicants must", "Users should",
              "Participants are required to", "Visitors must", "All staff must", "Tenants should"],
    "duty": ["sign in at the reception", "complete the safety training", "keep their receipts",
             "submit their own work", "follow the dress code", "read the terms carefully",
             "attend at least 80% of the lectures", "report changes of address", "update their contact details",
             "use the provided template", "cite all sources", "renew the permit annually"],
    "rule_intro": ["From now on,", "Starting next week,", "Effective immediately,", "As of January,",
                   "Going forward,", "Under the new policy,"],
    "rule": ["all invoices must include a purchase order number", "meetings will start at 9 am",
             "the library closes at 10 pm", "travel must be approved in advance",
             "late submissions will not be accepted", "parking requires a valid permit",
             "remote work requests go through your manager", "lab coats are mandatory"],
    "answer_task": ["Answer all questions", "Show your work", "Write your name on every page",
                    "Respond only in the space provided", "Summarize the article in 200 words",
                    "Translate the passage into Spanish", "Begin each answer on a new page",
                    "Explain your reasoning briefly", "Rate each item from 1 to 5", "Ignore question 4, it was cancelled",
                    "Ignore the first row of the table", "Disregard the earlier announcement about room changes",
                    "Use only a pencil", "List three examples", "Circle the correct answer"],
    "eval_subject": ["the proposal", "the candidate", "the submission", "the vendor's bid", "the prototype",
                     "her thesis", "the application", "the pilot project", "the restaurant", "the new policy"],
    "eval_verdict": ["was rated highly by the committee", "received excellent reviews",
                     "was considered the strongest option", "met all requirements", "was accepted with minor revisions",
                     "was judged exceptionally well prepared", "scored above average", "was recommended for funding"],
    "please": ["Please", "Kindly", "Remember to", "Make sure to", "Don't forget to", "Be sure to"],
    "polite_act": ["read the instructions carefully", "keep this receipt for your records",
                   "contact the help desk if you have questions", "bring a valid photo ID",
                   "arrive ten minutes early", "check your spam folder", "save your work frequently",
                   "fill in all required fields", "note the deadline in your calendar", "confirm your attendance"],
    "med": ["Take one tablet twice daily with food", "Do not exceed the stated dose",
            "Store below 25 degrees", "Consult your doctor if symptoms persist",
            "Shake well before use", "Keep out of reach of children"],
    "legal": ["The licensee shall not reverse engineer the software",
              "This agreement is governed by the laws of France",
              "The tenant agrees to pay rent on the first day of each month",
              "Either party may terminate this contract with 30 days notice",
              "The user acknowledges that the product is provided as is",
              "Confidential information must not be disclosed to third parties"],
    "ai_context": ["The AI model", "Our chatbot", "The virtual assistant", "The recommendation engine",
                   "The language model", "The spam filter"],
    "ai_desc": ["was trained on public data", "answers questions about orders", "can be disabled in settings",
                "achieved 92% accuracy on the benchmark", "is available in twelve languages",
                "summarizes meeting notes automatically", "flags suspicious transactions for review"],
}

# Course / assignment FAQ text: first-person questions, compile/test rules, and policies ABOUT AI use.
# These produced most false flags on real assignment PDFs, so they get their own families.
SLOTS.update({
    "faq_q": ["Do I", "Can I", "Should I", "May I", "Must I", "Am I allowed to", "Is it okay to", "Do we need to",
              "Can we", "What happens if I"],
    "faq_act": ["print with endl or a newline", "use the STL containers", "submit a zip file",
                "use global variables", "add extra helper functions", "change the header file",
                "work with a partner", "use code from the lecture slides", "submit late",
                "include the Makefile", "rename the main file", "use Python instead of C++",
                "upload a PDF instead of a Word file", "write comments in Turkish", "use recursion here"],
    "faq_ans": ["Yes, as long as the output matches exactly.", "No, please follow the given interface.",
                "Then it does not compile on the lab machines.", "Only if you cite the source.",
                "Check the announcement on the course page.", "Ask your teaching assistant first.",
                "The grader only checks the final output.", "No, late submissions lose points."],
    "server": ["dijkstra.cs.bilkent.edu.tr", "the lab machines", "the department server", "the grading server",
               "the course Docker image", "ix.cs.example.edu"],
    "compiler": ["g++ -std=c++17 -O2", "gcc -Wall -Wextra", "javac", "python3", "make", "clang++ -std=c++20",
                 "cmake --build build"],
    "ai_policy": ["Use of generative AI tools must be disclosed in your report.",
                  "You may use AI assistants for brainstorming but not for writing code.",
                  "Submissions generated by ChatGPT or similar tools will be treated as plagiarism.",
                  "For questions about academic integrity, see the course home page and the Honor Code.",
                  "The university's guideline on generative artificial intelligence applies to this course.",
                  "If you used an AI tool, state which one and how in the appendix.",
                  "AI-generated text detectors may be used during grading.",
                  "Students are responsible for any content produced with AI assistance."],
    "integrity": ["Do not share your solutions online.", "Discussing ideas is fine, but write your own code.",
                  "Copying code from classmates is a violation of the Honor Code.",
                  "All submissions are checked with plagiarism detection software."],
})

TEMPLATES = [
    "{faq_q} {faq_act}?", "{faq_q} {faq_act}? {faq_ans}", "Q: {faq_q} {faq_act}? A: {faq_ans}",
    "We compile your code with {compiler} on {server}, together with our tests.",
    "Your program must compile with {compiler} on {server}.",
    "If it works on your laptop but not on {server}, then it does not work.",
    "{server} is always right.", "Test your submission on {server} before the deadline.",
    "{ai_policy}", "{integrity}", "{ai_policy} {integrity}",
    "{verb_submit} {thing} {dest} {when}.", "{verb_submit} {thing} {when}.",
    "{dev_action} {dev_cond}.", "{dev_action}.", "{dev_action}, then {dev_lower}.",
    "{cook}.", "{cook}, then {cook_lower}.",
    "{safety}.", "{safety} at all times.",
    "{neg} {neg_act}.", "{modal} {duty}.", "{modal} {duty} {when}.",
    "{rule_intro} {rule}.", "{answer_task}.", "{answer_task} {when}.",
    "{eval_subject_cap} {eval_verdict}.", "{please} {polite_act}.",
    "{med}.", "{legal}.", "{ai_context} {ai_desc}.",
]

FRAGMENTS = [
    "{n}. {Heading} {dots} {p}", "{Heading} {dots} {p}", "{n}.{m} {Heading}", "Chapter {n}: {Heading}",
    "Table {n}: {Heading}", "Figure {n}. {Heading}", "- {Heading}", "• {Heading}", "{Heading}",
    "Page {p} of {q}", "{date}", "Version {n}.{m} – {date}", "{code}", "{n} {Heading} {p} {q}",
]
HEADINGS = ["Introduction", "Installation", "Data logging", "Frequently Asked Questions", "Disclaimer of warranty",
            "Terms of service", "Privacy policy", "Methods", "Results", "Discussion", "References", "Appendix",
            "Safety instructions", "Getting started", "Troubleshooting", "Grading", "Submission", "Overview",
            "Class Hierarchy", "Member Functions", "Severability", "Indemnification", "Learning outcomes"]
CODE = ["def __init__(self, a, b): self.a = a", "for i in range(10): print(i)", "return x + y",
        "int main() { return 0; }", "x = np.zeros((3, 3))", "SELECT name FROM users WHERE id = 1;",
        "if (err) { throw err; }", "import numpy as np", "class Rational:", "self.items.append(item)",
        "assign y = sel ? a : b;", "module mux2(input a, b, s, output y);"]


def _fill(t: str, rng: random.Random) -> str:
    def rep(m):
        key = m.group(1)
        if key == "dev_lower":
            v = rng.choice(SLOTS["dev_action"]); return v[0].lower() + v[1:]
        if key == "cook_lower":
            v = rng.choice(SLOTS["cook"]); return v[0].lower() + v[1:]
        if key == "eval_subject_cap":
            v = rng.choice(SLOTS["eval_subject"]); return v[0].upper() + v[1:]
        return rng.choice(SLOTS[key])
    return re.sub(r"\{(\w+)\}", rep, t)


def generate(n_sent: int = 3000, n_frag: int = 800, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = {_fill(rng.choice(TEMPLATES), rng) for _ in range(n_sent * 3)}
    sents = sorted(out)
    rng.shuffle(sents)
    frags = set()
    while len(frags) < n_frag:
        t = rng.choice(FRAGMENTS)
        frags.add(t.format(n=rng.randint(1, 12), m=rng.randint(1, 9), Heading=rng.choice(HEADINGS),
                           dots="." * rng.randint(8, 60), p=rng.randint(1, 40), q=rng.randint(40, 90),
                           date=f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/20{rng.randint(10, 26)}",
                           code=rng.choice(CODE)))
    return sents[:n_sent] + sorted(frags)
