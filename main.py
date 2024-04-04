# file usage imports
import sys
import os
import io
import json

# UI imports
from PyQt6 import uic
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog

# OCR import
from google.cloud import vision

# NER imports
import spacy
from spacy.tokens import DocBin

# text processing import
import regex as re

# iteration visualisation import
from tqdm import tqdm


class Main(QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        # Load UI structure
        self.ui = uic.loadUi('ui/Main.ui', self)

        # Connect UI buttons to respective function
        self.pushButton_reset.clicked.connect(self.reset_ui)
        self.pushButton_file.clicked.connect(self.file_dialog)

        # Link Key to Google Cloud Vision API
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './api_key/accounting-372210-2b903c0174ea.json'

        # Path to the NER-model folder to use for recognition
        self.ner_model = './ner_model/model_3'

        # single use methods that can be uncommented in necessary
        # self.save_recognized_text(file_size=1000, coordinates=True)
        # self.convert_training_data()
        # self.create_main_ner_model()
        # self.test_ner_model()

    """
    Resets the UI to default settings
    """
    def reset_ui(self):
        # reset the label texts
        self.label_company.setText('Unternehmen: ')
        self.label_reference.setText('Buchungsnummer: ')
        self.label_total.setText('Total: ')

        # reset the preview of the invoice image
        self.imageView.setPixmap(QPixmap())

        # enable / disable buttons for usability
        self.pushButton_reset.setEnabled(False)
        self.pushButton_file.setEnabled(True)

    """
    returns the text with annotations of given image
    """
    @staticmethod
    def detect_text(file_path: str):
        # instantiates a new client
        client = vision.ImageAnnotatorClient()

        # opens the file to the given path
        with io.open(file_path, 'rb') as image_file:
            content = image_file.read()

        # creates a compatible image for Google Cloud Vision API
        image = vision.Image(content=content)

        # return the recognized text of the given image
        response = client.document_text_detection(image=image)
        document = response.text_annotations
        return document

    """
    processes the text into a single line without annotations and add coordinates after each word if necessary
    """
    @staticmethod
    def preprocess_text(document, coordinates=False):
        out = str(document)
        if len(out) > 0:  # checks if there is any recognized text
            # transforms text to a single line
            out = out.replace('\n', ' ')
            out = out.replace('\\n', ' ')
            out = out.split('description: \"', 1)[1]
            out = out.split('\" bounding_poly', 1)[0]
            if coordinates:  # checks parameter if coordinates should be added after each word
                out = []
                for text in document:
                    # extracts corners of the words bounding box
                    vertices = [('({},{})'.format(vertex.x, vertex.y))
                                for vertex in text.bounding_poly.vertices]
                    out.append(text.description + str(vertices[0]))
                out.pop(0)
                out = ' '.join(out)
        return out

    """
    opens a dialog window to select an invoice image out of the file browser
    """
    def file_dialog(self):
        # opens dialog window
        file, check = QFileDialog.getOpenFileName(None, 'QFileDialog.getOpenFileName()',
                                                  '', 'JPEG (*.jpg);;PNG (*.png)')
        if check:  # if a file was chosen
            # reset the UI to default
            self.reset_ui()

            # set preview of invoice image
            self.imageView.setPixmap(QPixmap(file))

            # enable / disable buttons for usability
            self.pushButton_reset.setEnabled(True)
            self.pushButton_file.setEnabled(False)

            # start recognition and display pipeline
            self.execute_pipeline(file)

    """
    uses the given NER-model to classify the named entities;
    can classify each word on its own if necessary
    """
    def entity_recognition(self, text: [], block_processing=False):
        nlp = spacy.load(self.ner_model + '/model-best')  # opens the NER model
        print(text)
        entities = {'ORG': [], 'MONEY': [], 'INVOICE_NO': []}  # entities to be processed
        if block_processing:  # NER model processes the text word by word
            for word in text:
                doc = nlp(word)
                for ent in doc.ents:
                    if ent.label_ in entities.keys():
                        entities[ent.label_].append(ent.text)
                    if ent.label_ == 'CARDINAL':
                        entities['MONEY'].append(ent.text)
        else:  # NER model processes the entire text
            text = ' '.join(text)
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in entities.keys():
                    entities[ent.label_].append(ent.text)
                if ent.label_ == 'CARDINAL':
                    entities['MONEY'].append(ent.text)
        return entities

    """
    test method to check the accuracy of the currently used NER model;
    every named entity has its proper accuracy;
    accuracy is given for a 100 percent match, to contain more than the correct named entity and a failure rate
    """
    def test_ner_model(self):
        testdata = self.load_data(self.ner_model + "/validation_data.json")
        results = {
            "ORG": {
                "Accuracy": 0,
                "Accuracy_multi": 0,
                "Error-rate": 0
            },
            "INVOICE_NO": {
                "Accuracy": 0,
                "Accuracy_multi": 0,
                "Error-rate": 0
            },
            "MONEY": {
                "Accuracy": 0,
                "Accuracy_multi": 0,
                "Error-rate": 0
            }
        }
        for text, entity in tqdm(testdata):  # fills a dictionary with recognized named entities
            entities = {'ORG': [], 'MONEY': [], 'INVOICE_NO': []}
            for t in entity['entities']:
                entities[t[2]].append(text[t[0]:t[1]])
            recognized_entities = self.entity_recognition(text.split())
            # changes the respective accuracy
            for key in entities:
                if recognized_entities[key] == entities[key]:
                    results[key]['Accuracy'] = results[key]['Accuracy'] + 1
                elif any(item in entities[key] for item in recognized_entities[key]):
                    results[key]['Accuracy_multi'] = results[key]['Accuracy_multi'] + 1
                else:
                    results[key]['Error-rate'] = results[key]['Error-rate'] + 1

        # prints the accuracy by each named entity used in the model
        for key in results:
            if not results[key]['Accuracy'] == len(testdata):
                print(key)
                for value in results[key]:
                    print(value + ': ' + str(((results[key][value] / len(testdata)) * 100)) + '%')
                print('\n')

    """
    displays the recognized entities in the respective UI label
    """
    def display_entities(self, entities: dict):
        if entities.get('ORG'):  # company label
            self.label_company.setText(self.label_company.text() + ', '.join(entities.get('ORG')))
        if entities.get('MONEY'):  # money label
            self.label_total.setText(self.label_total.text() + ', '.join(entities.get('MONEY')))
        if entities.get('INVOICE_NO'):  # invoice number label
            self.label_reference.setText(self.label_reference.text() + ', '.join(entities.get('INVOICE_NO')))

    """
    saves the file path and the recognized text in a respective text files for further labeling
    """
    def save_recognized_text(self, file_size, coordinates=False):
        f = open("invoice.txt", "w")  # opens a text file to save the recognized text
        name = open('image.txt', 'w')  # opens a text file to save the image name
        count = 0  # counter to limit the amount of files to be processed
        for path in os.listdir('invoice'):
            if path.split('.')[1] == '{}'.format('jpg'):
                count += 1
                print('Files: ' + str(count))
                name.write(path + '\n')  # saves the image name
                file_path = './invoice/' + path
                out = self.detect_text(file_path)
                final = self.preprocess_text(out, coordinates)
                f.write(final + '\n' + '\n')  # saves the recognized text
            else:
                print('Config File')
            if count == file_size:  # stops if file size is reached
                name.close()
                f.close()
                print('Done!')
                return

    """
    loads a JSON file into a list
    """
    @staticmethod
    def load_data(file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    """
    help function to create a compatible dataset for spacy
    """
    @staticmethod
    def convert_to_spacy_file(train_data: []):
        model = spacy.blank('en')  # loads an empty english NER-model
        db = DocBin()
        for text, annotation in train_data:  # iterates over all labeled data entries
            doc = model.make_doc(text)
            ents = []
            for start, end, label in annotation['entities']:  # iterates over all labeled entities of a text
                span = doc.char_span(start, end, label=label, alignment_mode='contract')  # the start and end indices
                # of the given entity with the respective label are extracted
                if span is not None:  # check if a start-, endpoint and label were given and if so append entity
                    ents.append(span)
                doc.ents = ents
                db.add(doc)
        return db

    """
    converts the training and validation data into a spacy compatible format
    """
    def convert_training_data(self):
        training_data = self.convert_to_spacy_file(self.load_data(self.ner_model + '/training_data.json'))
        training_data.to_disk(self.ner_model + '/training_data.spacy')  # saves spacy compatible training data
        validation_data = self.convert_to_spacy_file(self.load_data(self.ner_model + '/validation_data.json'))
        validation_data.to_disk(self.ner_model + '/validation_data.spacy')  # saves spacy compatible validation data

    """
    creates a new NER-model based on the en_core_web_lg model to recognize INVOICE_NO, CARDINAL, MONEY, ORG 
    """
    def create_main_ner_model(self):
        main_model = spacy.load('en_core_web_lg')  # load large pretrained general model
        invoice_no_model = spacy.load(self.ner_model + '/model-best')  # load NER-model trained on invoice numbers
        invoice_no_model.replace_listeners('tok2vec', 'ner', ['model.tok2vec'])
        main_model.add_pipe('ner', source=invoice_no_model, name='invoice_no_ner', before='ner')
        main_model.to_disk(self.ner_model + '/main_model')  # saves the new pipeline

    """
    executes a pipeline to recognize text off a given image, classifies ORG, MONEY, INVOICE_NO and displays them in 
    the respective UI labels
    """
    def execute_pipeline(self, input_file: str):
        invoice_text = self.detect_text(input_file)  # detect text of a given image
        processed_invoice_text = self.preprocess_text(invoice_text)  # processes the text to a single line
        invoice_entities = self.entity_recognition(processed_invoice_text.split())  # recognition of named entities
        print(invoice_entities)
        self.display_entities(invoice_entities)  # displays named entities right of the models output


app = QApplication(sys.argv)
mainWindow = Main()
mainWindow.show()
sys.exit(app.exec())
