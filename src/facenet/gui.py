from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter.scrolledtext as tkscrolled
from facenet import tripless_lot_train
from utils import constants
from facenet.align import align_dataset_mtcnn
from facenet import classifier
import random


class Gui(Tk):
    def __init__(self):
        """
            Init GUI
        """
        super(Gui, self).__init__()
        self.title('Face recognition')
        self.minsize(constants.GUI_WIDTH, constants.GUI_HEIGHT)

        tab_control = ttk.Notebook(self)

        self.create_align_tab(tab_control)

        self.create_classifier_tab(tab_control)

        self.create_image_recognition_tab(tab_control)

        self.create_train_tab(tab_control)

        tab_control.pack(expan=1, fill='both')

    def create_train_tab(self, tab_control):
        """
            Create GUI for train model tab
        """
        train_tab = ttk.Frame(tab_control)
        # Data dir selection
        train_tab_data_dir_label = ttk.Label(train_tab, text="Data Dir : ").place(x=20, y=20)
        train_tab_data_dir_entry = ttk.Entry(train_tab, width=70)
        train_tab_data_dir_entry.place(x=110, y=20)  # have to place in a new line for use insert
        train_tab_data_dir_browse_button = ttk.Button(train_tab, text="Select Dir", command=lambda: self.select_dir(train_tab_data_dir_entry)).place(x=550, y=20)

        # Log dir selection
        train_tab_log_dir_label = ttk.Label(train_tab, text="Log Dir : ").place(x=20, y=50)
        train_tab_log_dir_entry = ttk.Entry(train_tab, width=70)
        train_tab_log_dir_entry.place(x=110, y=50)  # have to place in a new line for use insert
        train_tab_log_dir_browse_button = ttk.Button(train_tab, text="Select Dir", command=lambda: self.select_dir(train_tab_log_dir_entry)).place(x=550, y=50)

        # Model output dir selection
        train_tab_model_out_dir_label = ttk.Label(train_tab, text="Model Out Dir : ").place(x=20, y=80)
        train_tab_model_out_dir_entry = ttk.Entry(train_tab, width=70)
        train_tab_model_out_dir_entry.place(x=110, y=80)  # have to place in a new line for use insert
        train_tab_model_out_dir_browse_button = ttk.Button(train_tab, text="Select Dir", command=lambda: self.select_dir(train_tab_model_out_dir_entry)).place(x=550, y=80)

        # Batch size
        train_tab_batch_size_label = ttk.Label(train_tab, text="Batch Size : ").place(x=20, y=110)
        train_tab_batch_size_entry = ttk.Entry(train_tab, width=20)
        train_tab_batch_size_entry.place(x=110, y=110)

        # Epoch size
        train_tab_epoch_size_label = ttk.Label(train_tab, text="Epoch Size : ").place(x=20, y=140)
        train_tab_epoch_size_entry = ttk.Entry(train_tab, width=20)
        train_tab_epoch_size_entry.place(x=110, y=140)

        train_tab_start_train_button = ttk.Button(train_tab, text="Start Train", command=lambda: self.start_train_model(train_tab_data_dir_entry,
                                                                                                                        train_tab_log_dir_entry,
                                                                                                                        train_tab_model_out_dir_entry,
                                                                                                                        train_tab_batch_size_entry,
                                                                                                                        train_tab_epoch_size_entry))
        train_tab_start_train_button.place(x=constants.GUI_WIDTH - 100, y=constants.GUI_HEIGHT - 75)
        tab_control.add(train_tab, text="  Train Model  ")

    def create_align_tab(self, tab_control):
        """
            Create GUI for align data tab
        """
        # Align data tab
        align_data_tab = ttk.Frame(tab_control)
        align_data_tab_data_dir_label = ttk.Label(align_data_tab, text="Data Dir : ").place(x=20, y=20)
        align_data_tab_data_dir_entry = ttk.Entry(align_data_tab, width=70)
        align_data_tab_data_dir_entry.place(x=110, y=20)
        align_data_tab_data_dir_browse_button = ttk.Button(align_data_tab, text="Select Dir", command=lambda: self.select_dir(align_data_tab_data_dir_entry)).place(x=550, y=20)

        # output dir selection
        align_data_tab_output_dir_label = ttk.Label(align_data_tab, text="Output Dir : ").place(x=20, y=50)
        align_data_tab_output_dir_entry = ttk.Entry(align_data_tab, width=70)
        align_data_tab_output_dir_entry.place(x=110, y=50)
        align_data_tab_output_dir_browse_button = ttk.Button(align_data_tab, text="Select Dir", command=lambda: self.select_dir(align_data_tab_output_dir_entry)).place(x=550, y=50)

        # Text Area
        # Horizontal (x) Scroll bar
        xscrollbar = Scrollbar(align_data_tab, orient=HORIZONTAL)
        xscrollbar.place(x=20, y=320, width=530)

        # Vertical (y) Scroll Bar
        yscrollbar = Scrollbar(align_data_tab, orient=VERTICAL)
        yscrollbar.place(x=550, y=110, height=210)

        align_data_textarea = Text(align_data_tab, wrap=NONE, xscrollcommand=xscrollbar.set, yscrollcommand=yscrollbar.set)
        align_data_textarea.place(x=20, y=110, width=530, height=210)
        xscrollbar.config(command=align_data_textarea.xview)
        yscrollbar.config(command=align_data_textarea.yview)

        # Align button
        align_data_start_train_button = ttk.Button(align_data_tab, text="Start Align",
                                                   command=lambda: self.start_align_data(align_data_tab_data_dir_entry, align_data_tab_output_dir_entry, align_data_textarea))
        align_data_start_train_button.place(x=110, y=80, width=425)

        tab_control.add(align_data_tab, text="  Align Data  ")

    def create_classifier_tab(self, tab_control):
        """
            Create GUI for classifier Tab
        """
        # Classifier data tab
        classifier_tab = ttk.Frame(tab_control)
        classifier_tab_data_dir_label = ttk.Label(classifier_tab, text="Data Dir : ").place(x=20, y=20)
        classifier_tab_data_dir_entry = ttk.Entry(classifier_tab, width=70)
        classifier_tab_data_dir_entry.place(x=110, y=20)
        classifier_tab_data_dir_browse_button = ttk.Button(classifier_tab, text="Select Dir", command=lambda: self.select_dir(classifier_tab_data_dir_entry)).place(x=550, y=20)

        # output dir selection
        classifier_tab_model_dir_label = ttk.Label(classifier_tab, text="Model Dir : ").place(x=20, y=50)
        classifier_tab_model_dir_entry = ttk.Entry(classifier_tab, width=70)
        classifier_tab_model_dir_entry.place(x=110, y=50)
        classifier_tab_model_dir_browse_button = ttk.Button(classifier_tab, text="Select Dir", command=lambda: self.select_dir(classifier_tab_model_dir_entry)).place(x=550, y=50)

        # output dir selection
        classifier_tab_output_dir_label = ttk.Label(classifier_tab, text="Output Dir : ").place(x=20, y=80)
        classifier_tab_output_dir_entry = ttk.Entry(classifier_tab, width=70)
        classifier_tab_output_dir_entry.place(x=110, y=80)
        classifier_tab_output_dir_browse_button = ttk.Button(classifier_tab, text="Select Dir", command=lambda: self.select_dir(classifier_tab_output_dir_entry)).place(x=550, y=80)

        # Text Area
        # Horizontal (x) Scroll bar
        xscrollbar = Scrollbar(classifier_tab, orient=HORIZONTAL)
        xscrollbar.place(x=20, y=320, width=530)

        # Vertical (y) Scroll Bar
        yscrollbar = Scrollbar(classifier_tab, orient=VERTICAL)
        yscrollbar.place(x=550, y=140, height=210)

        classifier_tab_textarea = Text(classifier_tab, wrap=NONE, xscrollcommand=xscrollbar.set, yscrollcommand=yscrollbar.set)
        classifier_tab_textarea.place(x=20, y=140, width=530, height=180)
        xscrollbar.config(command=classifier_tab_textarea.xview)
        yscrollbar.config(command=classifier_tab_textarea.yview)

        # Classifier button
        align_data_start_train_button = ttk.Button(classifier_tab, text="Start generate classifier",
                                                   command=lambda: self.start_classifier_data(classifier_tab_data_dir_entry, classifier_tab_model_dir_entry, classifier_tab_output_dir_entry, classifier_tab_textarea))
        align_data_start_train_button.place(x=110, y=110, width=425)

        tab_control.add(classifier_tab, text="  Generate Classifier  ")

    def create_image_recognition_tab(self, tab_control):
        """
            Create GUi for image recognition tab
        """
        image_recog_tab = ttk.Frame(tab_control)
        image_recog_tab_data_dir_label = ttk.Label(image_recog_tab, text="Image file").place(x=20, y=20)
        image_recog_tab_data_dir_entry = ttk.Entry(image_recog_tab, width=70)
        image_recog_tab_data_dir_entry.place(x=110, y=20)
        image_recog_tab_data_dir_browse_button = ttk.Button(image_recog_tab, text="Select File", command=lambda: self.select_file(image_recog_tab_data_dir_entry)).place(x=550, y=20)

        # Classifier button
        align_data_start_train_button = ttk.Button(image_recog_tab, text="Start Recognize", command=lambda: self.start_image_recognition(image_recog_tab_data_dir_entry, image_recog_tab))
        align_data_start_train_button.place(x=110, y=50, width=425)

        tab_control.add(image_recog_tab, text="  Recognition Image  ")

    def select_dir(self, entry: Entry):
        """
            Select directory and set the directory absolute path to entry
        """
        entry.delete(0, END)
        filename = filedialog.askdirectory()
        entry.insert(0, filename)

    def select_file(self, entry: Entry):
        """
            Select directory and set the directory absolute path to entry
        """
        entry.delete(0, END)
        filename = filedialog.askopenfilenames()
        entry.insert(0, filename)

    def start_train_model(self,
                          train_tab_data_dir_entry: Entry,
                          train_tab_log_dir_entry: Entry,
                          train_tab_model_out_dir_entry: Entry,
                          train_tab_batch_size_entry: Entry,
                          train_tab_epoch_size_entry: Entry):
        """
            Handle event train model
        """

        tripless_lot_train.main(train_tab_data_dir_entry.get(),
                                train_tab_log_dir_entry.get(),
                                train_tab_model_out_dir_entry.get(),
                                train_tab_batch_size_entry.get(),
                                train_tab_epoch_size_entry.get())
        return

    def start_align_data(self,
                         align_data_tab_data_dir_entry: Entry,
                         align_data_tab_output_dir_entry: Entry,
                         align_data_textarea: Text):
        """
            Handle event Align data
        """
        if (align_data_tab_data_dir_entry.get() == '') | (align_data_tab_output_dir_entry.get() == ''):
            align_data_textarea.delete('1.0', END)
            align_data_textarea.insert(END, 'The data dir and the output dir must not be empty.')
            return

        align_data_textarea.delete('1.0', END)
        align_data_textarea.insert(END, 'Align image is processing...\r\n')
        align_data_textarea.update()
        align_data_textarea.delete('1.0', END)
        nrof_images_total, nrof_successfully_aligned, face_detected_list = align_dataset_mtcnn.main(align_data_tab_data_dir_entry.get(), align_data_tab_output_dir_entry.get());

        align_data_textarea.insert(END, "{0} face detected in {1} images \r\n\r\n".format(nrof_successfully_aligned, nrof_images_total))
        for face_detected_path in face_detected_list:
            align_data_textarea.insert(END, face_detected_path)

        return

    def start_classifier_data(self,
                              classifier_tab_data_dir_entry: Entry,
                              classifier_tab_model_dir_entry: Entry,
                              classifier_tab_output_dir_entry: Entry,
                              classifier_textarea: Text):
        """
            Handle event classification data
        """
        if (classifier_tab_data_dir_entry.get() == '') | (classifier_tab_model_dir_entry.get() == '') | (classifier_tab_output_dir_entry.get() == ''):
            classifier_textarea.delete('1.0', END)
            classifier_textarea.insert(END, 'The data dir, model dir and the output dir must not be empty.')
            return

        classifier_textarea.delete('1.0', END)
        classifier_textarea.insert(END, 'Classifier is processing...\r\n')
        classifier_textarea.update()
        classifier_textarea.delete('1.0', END)
        classifier_file_name = classifier.main(classifier_tab_data_dir_entry.get(), classifier_tab_model_dir_entry.get(), classifier_tab_output_dir_entry.get())
        classifier_textarea.insert(END, 'Classifier is finished. The classifier file is stored at:\r\n\r\n')
        classifier_textarea.insert(END, classifier_file_name)
        return

    def start_image_recognition(self, image_recog_tab_data_dir_entry: Entry, image_recog_tab):
        """
            Handle image recognition event
        """

        return


if __name__ == '__main__':
    gui = Gui()
    gui.mainloop()
