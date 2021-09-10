from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt


class EdgeDetectionGUI:

    def __init__(self, master):

        # init params
        self.size = 600
        self.left_image = None
        self.right_image = None
        self.past = 3
        self.edge_image_exists = False
        self.edge_params = {
            'blur_kernel_size': 3,
            'min_val': 0,
            'max_val': 40
        }

        # make outer frame
        self.frame = Frame(master)
        self.frame.grid(row=0, column=0)

        # make options frame
        self.options_frame = Frame(self.frame, height=self.size//2, width=self.size//2)
        self.options_frame.grid(row=1, column=0, columnspan=4)

        # make sure we can define the width ourselves
        #self.options_frame.grid_propagate(False)

        # make sure the buttons actually go to the rows/columns we want them to
        #self.options_frame.grid_rowconfigure(0, weight=1)
        #self.options_frame.grid_rowconfigure(99, weight=1)
        self.options_frame.grid_columnconfigure(0, weight=1)
        self.options_frame.grid_columnconfigure(12, weight=1)

        # make load image button
        self.button1 = Button(self.options_frame, text="Load Image", command=self.load_image)
        self.button1.grid(row=2, column=4, columnspan=4)

        # leave space after top button
        self.space1 = Label(self.options_frame, text="")
        self.space1.grid(row=3, column=4, columnspan=4)

        # slider for gaussian blur kernel size
        self.slider1 = Scale(self.options_frame, from_=3, to=21, label='Blur Kernel Size', command=self.slider1_change, orient=HORIZONTAL) 
        self.slider1.grid(row=4, column=4, columnspan=4)
        
        # slider for canny min
        self.slider2 = Scale(self.options_frame, from_=0, to=400, resolution=5, label='Canny Min', command=self.slider2_change, orient=HORIZONTAL) 
        self.slider2.grid(row=5, column=4, columnspan=4)

        # slider for canny max
        self.slider3 = Scale(self.options_frame, from_=20, to=400, resolution=5, label='Canny Max', command=self.slider3_change, orient=HORIZONTAL) 
        self.slider3.grid(row=6, column=4, columnspan=4)

        # leave space before bottom button
        self.space2 = Label(self.options_frame, text="")
        self.space2.grid(row=8, column=4, columnspan=4)

        # make save image button
        self.button2 = Button(self.options_frame, text="Save Output", command=self.save_output)
        self.button2.grid(row=9, column=4, columnspan=4)
        
        # leave space after bottom button
        self.space3 = Label(self.options_frame, text="")
        self.space3.grid(row=12, column=4, columnspan=4)

        # frame for left image
        self.left_frame = Canvas(self.frame, height=self.size, width=self.size, borderwidth=2, relief=RIDGE)
        self.left_frame.grid(row=1, column=4, columnspan=4)

        # frame for right image
        self.right_frame = Canvas(self.frame, height=self.size, width=self.size, borderwidth=2, relief=RIDGE)
        self.right_frame.grid(row=1, column=8, columnspan=4)

    def load_image(self):
        self.filename = filedialog.askopenfilename(initialdir = ".",
                                                   title = "Select Image",
                                                   filetypes=[("All files","*.*")])

        if self.filename:
            img = cv2.imread(self.filename)
            self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            del img
            self.draw_edges()
            self.edge_image_exists = True

    def save_output(self):
        if self.edge_image_exists:
            # convert back to white edges for the .dilate() function to work
            imginv = cv2.bitwise_not(self.edge_image)
            # use .dilate() to thicken the edges
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            kernel = np.ones((3,3),np.uint8)
            dilated = cv2.dilate(imginv, kernel, iterations=1)
            # save the transparency channel alpha
            alpha = np.where(dilated > 0, 255, 0)
            imginv2 = cv2.bitwise_not(alpha)
            # duplicate the grayscale edge_image to mimic the BGR image and finally add the transparency
            new_img = cv2.merge((imginv2, imginv2, imginv2, alpha))
            # make the background transparent with ax.set_rasterized(True)
            # NOTE: this was not documented well...
            fig, ax = plt.subplots(1,1)
            ax.set_rasterized(True)
            # remove axes so only the edges are visible
            plt.axis('off')
            # set the image to the figure and save it
            # NOTE: we don't actually need to show it, but i don't see another way...
            imgplot = plt.imshow(new_img)
            plt.savefig("fig.pdf")
            print(self.edge_params)

    def change_params(self, name, value):
        self.edge_params[name] = value
        self.draw_edges()

    def draw_edges(self):
        smoothedInput = cv2.GaussianBlur(self.img_rgb, 
                                         (self.edge_params['blur_kernel_size'], self.edge_params['blur_kernel_size']), 
                                         0)
        #smoothedInput = cv2.bilateralFilter(gray, 9, 75, 75) # better at preserving edges

        self.edge_image = cv2.Canny(smoothedInput,
                          self.edge_params['min_val'],
                          self.edge_params['max_val'],
                          3
        )
        self.edge_image = cv2.bitwise_not(self.edge_image)

        self.display_in_tkinter()

    def fix(self, n):
        n = int(n)
        if not n % 2:
            self.slider1.set(n+1 if n > self.past else n-1)
            self.past = self.slider1.get()

    def slider1_change(self, n):
        self.fix(n)
        self.edge_params['blur_kernel_size'] = self.slider1.get()
        self.draw_edges()

    def slider2_change(self, n):
        self.edge_params['min_val'] = self.slider2.get()
        self.draw_edges()

    def slider3_change(self, n):
        self.edge_params['max_val'] = self.slider3.get()
        self.draw_edges()

    def display_in_tkinter(self):
        # LEFT IMAGE
        self.imgtk1 = ImageTk.PhotoImage(image=Image.fromarray(self.img_rgb))
        # tkinter requires a PhotoImage object
        if self.left_image is None:
            self.left_image = self.left_frame.create_image(self.size//2 + 4, self.size//2 + 4, image=self.imgtk1)
        else:
            self.left_frame.itemconfig(self.left_image, image=self.imgtk1)

        # RIGHT IMAGE
        self.imgtk2 = ImageTk.PhotoImage(image=Image.fromarray(self.edge_image))
        # tkinter requires a PhotoImage object
        if self.right_image is None:
            self.right_image = self.right_frame.create_image(self.size//2 + 4, self.size//2 + 4, image=self.imgtk2)
        else:
            self.right_frame.itemconfig(self.right_image, image=self.imgtk2)


def main():
    root = Tk()
    root.title("Edge Detection GUI")
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes,'-topmost', False)
    gui = EdgeDetectionGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
