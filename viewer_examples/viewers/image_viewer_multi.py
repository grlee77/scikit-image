from skimage import data
#from skimage.viewer import ImageViewer
import numpy as np
from skimage import io, img_as_float
from skimage.viewer.qt import QtGui, QtCore
from skimage.viewer.qt.QtCore import Qt, Signal
from skimage.viewer import utils
from skimage.util.dtype import dtype_range
from skimage.viewer.plugins.base import Plugin
from skimage.viewer.widgets import Slider
from skimage.viewer.viewers.core import BlitManager, EventManager
from skimage.viewer.utils import dialogs


from functools import partial
from matplotlib import lines

# if False:  #see viewer.canvastools.linetool, etc...
    
# 	line_props = None
#     props = dict(color='r', linewidth=1, alpha=0.4) #, solid_capstyle='butt')
#     props.update(line_props if line_props is not None else {})
#     x = (0, 0)
#     y = (0, 0)    
#     self.lines = []
#     #self.artists = []
#     line = lines.Line2D(x, y, visible=True, animated=True, **props)
#     self.lines.append(line)
#     self.axes[v].add_line(line)
#     #self.artists.append(self._line)
#     #self._line.set_data(np.transpose(pts))


class ImageViewer(QtGui.QMainWindow):
    """Viewer for displaying images.

    This viewer is a simple container object that holds a Matplotlib axes
    for showing images. `ImageViewer` doesn't subclass the Matplotlib axes (or
    figure) because of the high probability of name collisions.

    Parameters
    ----------
    image : array
        Image being viewed.

    Attributes
    ----------
    canvas, fig, ax : Matplotlib canvas, figure, and axes
        Matplotlib canvas, figure, and axes used to display image.
    image : array
        Image being viewed. Setting this value will update the displayed frame.
    original_image : array
        Plugins typically operate on (but don't change) the *original* image.
    plugins : list
        List of attached plugins.

    Examples
    --------
    >>> from skimage import data
    >>> image = data.coins()
    >>> viewer = ImageViewer(image) # doctest: +SKIP
    >>> viewer.show()               # doctest: +SKIP

    """

    dock_areas = {'top': Qt.TopDockWidgetArea,
                  'bottom': Qt.BottomDockWidgetArea,
                  'left': Qt.LeftDockWidgetArea,
                  'right': Qt.RightDockWidgetArea}

    # Signal that the original image has been changed
    original_image_changed = Signal(np.ndarray)

    def __init__(self, image, useblit=True, image_labels=None, ortho_viewer = False, active_image_index = 0):
        # Start main loop
        utils.init_qtapp()
        super(ImageViewer, self).__init__()

        #TODO: Add ImageViewer to skimage.io window manager

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Image Viewer")

        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('Open file', self.open_file,
                                 Qt.CTRL + Qt.Key_O)
        self.file_menu.addAction('Save to file', self.save_to_file,
                                 Qt.CTRL + Qt.Key_S)
        self.file_menu.addAction('Quit', self.close,
                                 Qt.CTRL + Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.main_widget = QtGui.QWidget()
        self.setCentralWidget(self.main_widget)

        def _is_color(img):
        	if img.shape[-1] in [3,4]: 
        		return True
        	else:
        		return False

        nviews=1
        if isinstance(image, Plugin):
            plugin = image
            image = plugin.filtered_image
            plugin.image_changed.connect(self._update_original_image)
            # When plugin is started, start
            plugin._started.connect(self._show)
        elif isinstance(image, (list,tuple,set)):
            if ortho_viewer:
                raise ValueError("list input not supported when ortho_viewer = True")
            #from copy import deepcopy
            nviews = len(image)
            if (image[0].ndim == 2) or ((image[0].ndim ==3) and _is_color(image[0])):
                self.images = image
                #self.volumes = self.images
                self.data_is_3D = False
            else:  #use separate volumes to retain full 3D volumes
                self.data_is_3D = True
                self.volumes = image  
                self.images = []
                self.current_frames = []
                for vol in self.volumes:
                    nimg = vol.shape[-1]
                    frame = int(np.floor(nimg/2.))-1
                    frame = max(frame,0)
                    self.current_frames.append(frame)
                    self.images.append(vol[...,frame])
            image = self.images[0]
        else:
            if ortho_viewer:
                if not (image.ndim - _is_color(image)) == 3:
                    raise ValueError("Must provide 3D data as input when ortho_viewer = True")

            if (image.ndim == 2) or ((image.ndim ==3) and _is_color(image)):
                self.data_is_3D = False
                self.images = [image,]
            else:
                self.data_is_3D = True
                if ortho_viewer:
	                nviews = 3
	                mid_frames = np.floor(np.asarray(image.shape)/2.)-1
	                mid_frames = np.maximum(mid_frames,[0,0,0]).astype(np.intp)
	                self.current_frames = list(mid_frames)
	                self.images = [image[mid_frames[0],:,:], 
	                				   image[:,mid_frames[1],:], 
	                				   image[:,:,mid_frames[2]]]
                	self.volumes = [image, image, image]
                else:
	                nimg = image.shape[-1]
	                frame = int(np.floor(nimg/2.))-1
	                frame = max(frame,0)
	                self.current_frames = [frame, ]
	                self.images = [image[...,frame],]
                	self.volumes = [image,]
                image = self.images[0]

        self.image_labels = image_labels
#        if image_labels is not None:
#            if len(image_labels) != len(self.images)
#                raise ValueError("")
        #active image control which panel any attached plugins will operate upon
        if active_image_index > nviews:
            raise ValueError("active_image cannot exceed the number of images in image list")
        else:
        	self.active_image_index = active_image_index
        	active_image = self.images[active_image_index]

        ndims_list = []
        self.is_color = []
        for tmp in self.images:
        	ndim = tmp.ndim
        	#RGB or RGBA: treat as ndim-1
        	if _is_color(tmp): 
        		ndim -= 1
        		self.is_color.append(True)
        	else:
        		self.is_color.append(False)

        	ndims_list.append(ndim)
        if np.any(np.less(ndims_list, 2)) or np.any(np.greater(ndims_list, 3)):
        	raise ValueError("all images/volumes in list must be either 2D or 3D")
        if len(set(ndims_list)) > 1:
        	raise ValueError("all images in list must have the same number of dimensions")

        self._tool_lists = []
        
        
        self.useblit = useblit

        self.figures = []
        self.axes = []
        self.canvases = []
        self._image_plots = []
        self._event_managers = []
        self._labels = []
        self.slider_list = []       
        self.sub_layouts = []
        if self.useblit:
            self._blit_managers = []
        
        self.plugins = []

        self.layout = QtGui.QVBoxLayout(self.main_widget)

        status_bar = self.statusBar()
        self.status_message = status_bar.showMessage
        sb_size = status_bar.sizeHint()

        self.fig_layout = QtGui.QHBoxLayout(self.main_widget)

        canvas_heights = []
        canvas_widths = []

        for v in range(nviews):
            cs_height, other_height, w = self.generate_figlayout(frame=v, 
                                                                 ortho_viewer=ortho_viewer, 
                                                                 add_slider = True)
            canvas_heights.append(cs_height) #sub_layout_height)
            canvas_widths.append(w)

        self._update_original_image(active_image)

        self.layout.addLayout(self.fig_layout)

        #will use largest height as the figure height
        canvas_heights = np.asarray(canvas_heights)
        relative_heights = canvas_heights/float(canvas_heights.max())
        #scale up widths accordingly
        canvas_widths = np.asarray(canvas_widths)/relative_heights
        #round widths to integer number of pixels
        canvas_widths = np.round(canvas_widths).astype(np.intp)

#        self.layout.addLayout(self.fig_layout)
        self.resize(canvas_widths.sum(), canvas_heights.max() + 
                                         other_height + 
                                         sb_size.height())

        #self.connect_event('motion_notify_event', self._update_status_bar, i=0)

    def __add__(self, plugin):
        """Add plugin to ImageViewer"""
        plugin.attach(self)
        self.original_image_changed.connect(plugin._update_original_image)

        if plugin.dock:
            location = self.dock_areas[plugin.dock]
            dock_location = Qt.DockWidgetArea(location)
            dock = QtGui.QDockWidget()
            dock.setWidget(plugin)
            dock.setWindowTitle(plugin.name)
            self.addDockWidget(dock_location, dock)

            horiz = (self.dock_areas['left'], self.dock_areas['right'])
            dimension = 'width' if location in horiz else 'height'
            self._add_widget_size(plugin, dimension=dimension)

        return self

    def _add_widget_size(self, widget, dimension='width'):
        widget_size = widget.sizeHint()
        viewer_size = self.frameGeometry()

        dx = dy = 0
        if dimension == 'width':
            dx = widget_size.width()
        elif dimension == 'height':
            dy = widget_size.height()

        w = viewer_size.width()
        h = viewer_size.height()
        self.resize(w + dx, h + dy)

    def open_file(self, filename=None):
        """Open image file and display in viewer."""
        if filename is None:
            filename = dialogs.open_file_dialog()
        if filename is None:
            return
        image = io.imread(filename)
        self._update_original_image(image)

    def _update_original_image(self, image):
        self.original_image = image     # update saved image
        self.image = image.copy()       # update displayed image
        self.original_image_changed.emit(image)

    def save_to_file(self, filename=None):
        """Save current image to file.

        The current behavior is not ideal: It saves the image displayed on
        screen, so all images will be converted to RGB, and the image size is
        not preserved (resizing the viewer window will alter the size of the
        saved image).
        """
        if filename is None:
            filename = dialogs.save_file_dialog()
        if filename is None:
            return
        if len(self.ax.images) == 1:
            io.imsave(filename, self.image)
        else:
            underlay = mpl_image_to_rgba(self.ax.images[0])
            overlay = mpl_image_to_rgba(self.ax.images[1])
            alpha = overlay[:, :, 3]

            # alpha can be set by channel of array or by a scalar value.
            # Prefer the alpha channel, but fall back to scalar value.
            if np.all(alpha == 1):
                alpha = np.ones_like(alpha) * self.ax.images[1].get_alpha()

            alpha = alpha[:, :, np.newaxis]
            composite = (overlay[:, :, :3] * alpha +
                         underlay[:, :, :3] * (1 - alpha))
            io.imsave(filename, composite)

    def closeEvent(self, event):
        self.close()

    def _show(self, x=0):
        self.move(x, 0)
        for p in self.plugins:
            p.show()
        super(ImageViewer, self).show()
        self.activateWindow()
        self.raise_()

    def show(self, main_window=True):
        """Show ImageViewer and attached plugins.

        This behaves much like `matplotlib.pyplot.show` and `QWidget.show`.
        """
        self._show()
        if main_window:
            utils.start_qtapp()
        return [p.output() for p in self.plugins]

    def redraw(self,i=0):
        if self.useblit:
            self._blit_managers[i].redraw()
        else:
            self.canvases[i].draw_idle()

    @property
    def image(self):
        return self._img

    @image.setter
    def image(self, image):
        self._img = image
        self.set_image(image, i=self.active_image_index)

    def set_image(self, image, i=0):
    	"""image setter for any of the images in the list"""
    	if i>len(self.images):
    		raise ValueError("Invalid Index")
    	self.images[i] = image
        #self._img = image
        utils.update_axes_image(self._image_plots[i], image)

        # update display (otherwise image doesn't fill the canvas)
        h, w = image.shape[:2]
        self.axes[i].set_xlim(0, w)
        self.axes[i].set_ylim(h, 0)

        # update color range
        clim = dtype_range[image.dtype.type]
        if clim[0] < 0 and image.min() >= 0:
            clim = (0, clim[1])
        self._image_plots[i].set_clim(clim)

        if self.useblit:
            self._blit_managers[i].background = None

        self.redraw(i=i)

    def reset_image(self):
        self.image = self.original_image.copy()

    def connect_event(self, event, callback, i=0):
        """Connect callback function to matplotlib event and return id."""
        cid = self.canvases[i].mpl_connect(event, callback)
        return cid

    def disconnect_event(self, callback_id, i=0):
        """Disconnect callback by its id (returned by `connect_event`)."""
        self.canvases[i].mpl_disconnect(callback_id)

    def _update_status_bar(self, event, i=0):
        if event.inaxes and event.inaxes.get_navigate():
            self.status_message(self._format_coord(event.xdata, event.ydata, i=i))
        else:
            self.status_message('')

    def add_tool(self, tool, i=0):
        if self.useblit:
            self._blit_managers[i].add_artists(tool.artists)
        self._tool_lists[i].append(tool)
        self._event_managers[i].attach(tool)

    def remove_tool(self, tool, i=0):
        if self.useblit:
            self._blit_managers[i].remove_artists(tool.artists)
        self._tool_lists[i].remove(tool)
        self._event_managers[i].detach(tool)

    def _format_coord(self, x, y, i=0):
        # callback function to format coordinate display in status bar
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "%4s @ [%4s, %4s]" % (self.images[i][y, x], x, y)
        except IndexError:
            return ""

    def update_index(self, name, index, i, axis):
        index = int(round(index))

        if index == self.current_frames[i]:
            return

        # clip index value to collection limits
        num_images = self.volumes[i].shape[axis]
        index = max(index, 0)
        index = min(index, num_images - 1)

        self.current_frames[i] = index
        self.slider_list[i].val = index
        if (axis == -1) or (axis == 2):
        	self.set_image(self.volumes[i][...,index], i=i)
        elif (axis == 0):
        	self.set_image(self.volumes[i][index,...], i=i)
        elif (axis == 1):
        	self.set_image(self.volumes[i][:,index,...], i=i)
        elif (axis == 2):
        	self.set_image(self.volumes[i][:,:,index,...], i=i)
        else:
        	raise ValueError("unsupported axis")
        pass

    def generate_figlayout(self, frame=0, ortho_viewer = False, add_slider=True):

        sub_layout = QtGui.QVBoxLayout(self.main_widget)

        if frame<len(self.images):
            fig, ax = utils.figimage(self.images[frame])
        else:
            fig, ax = utils.figimage(np.zeros_like(self.images[-1]))

        canvas = fig.canvas
        canvas.setParent(self)
        ax.autoscale(enable=False)

        cs_size = canvas.sizeHint()

        self.figures.append(fig)
        self.axes.append(ax)
        self.canvases.append(canvas)
        self._image_plots.append(ax.images[0])
        self._event_managers.append(EventManager(ax))
        if self.useblit:
            self._blit_managers.append(BlitManager(ax))

        if self.image_labels is not None:
            label = QtGui.QLabel()
            label.setText(self.image_labels[frame])
            label.setAlignment(QtCore.Qt.AlignCenter)
            self._labels.append(label)
            label_height = label.sizeHint().height()
            sub_layout.addWidget(label)
        else:
            label_height = 0
        sub_layout.addWidget(canvas)

        if self.data_is_3D:
            num_images = self.volumes[frame].shape[-1]
            if ortho_viewer:
                if False: #testing overlaying lines
                    frame_list = [0,1,2]
                    other_frames = frame_list.remove(frame)                    
                    line_props = None
                    props = dict(color='r', linewidth=1, alpha=0.4) #, solid_capstyle='butt')
                    props.update(line_props if line_props is not None else {})
                    x = (0, self.volumes[other_frames[0]].shape[1])
                    y = (self.current_frames[other_frames[0]] + 0.5, self.current_frames[other_frames[0]] + 0.5)
                    self.lines = []
                    #self.artists = []
                    line = lines.Line2D(x, y, visible=True, animated=False, **props)
                    self.lines.append(line)

                    y2 = (0, self.volumes[other_frames[1]].shape[1])
                    x2 = (self.current_frames[other_frames[1]] + 0.5, self.current_frames[other_frames[1]] + 0.5)
                    line2 = lines.Line2D(x2, y2, visible=True, animated=False, **props)
                    #self.lines.append(line2)

                    self.axes[0].add_line(line)
                    self.axes[0].add_line(line2)
                #self.artists.append(self._line)

            if add_slider is not None:
                if ortho_viewer:
                    slice_axis = 0
                else:
                    slice_axis = -1
                slider_kws = {}
                slider_kws = dict(value = self.current_frames[frame], 
                                  low = 0, 
                                  high = num_images - 1,
                                  update_on = 'move',
                                  orientation = 'horizontal',
                                  value_type = 'int')
                slider_kws['callback'] = partial(self.update_index, 
                                                 i=frame, 
                                                 axis=slice_axis)
                slider = Slider('frame', **slider_kws)
                sub_layout.addWidget(slider)
                self.slider_list.append(slider)
                slider_height = slider.sizeHint().height()
            else:
                slider_height = 0

        else:
            slider_height = 0

        self._tool_lists.append([])
        self.connect_event('motion_notify_event', self._update_status_bar, i=frame)

        self.sub_layouts.append(sub_layout)
        self.fig_layout.addLayout(sub_layout)

        canvas_height = cs_size.height()
        width = cs_size.width()
        other_height = label_height + slider_height

        return canvas_height, other_height, width


import nibabel as nib
class CustomPlugin(Plugin):
    """ Kludge to substitute a different argument instead of 
    image_viewer.original_image as the first argument to the filter 
    """
    def __init__(self, first_argument_to_filter=None, **kwargs):
        super(CustomPlugin, self).__init__(**kwargs)
        self.first_argument_to_filter = first_argument_to_filter
    
    def attach(self, image_viewer):
        """Attach the plugin to an ImageViewer.

        Note that the ImageViewer will automatically call this method when the
        plugin is added to the ImageViewer. For example::

            viewer += Plugin(...)

        Also note that `attach` automatically calls the filter function so that
        the image matches the filtered value specified by attached widgets.
        """
        self.setParent(image_viewer)
        self.setWindowFlags(Qt.Dialog)

        self.image_viewer = image_viewer
        self.image_viewer.plugins.a
        ppend(self)
        #TODO: Always passing image as first argument may be bad assumption.
        if self.first_argument_to_filter == None:
            self.arguments = [self.image_viewer.original_image]
        else:
            self.arguments = [self.first_argument_to_filter]

        # Call filter so that filtered image matches widget values
        print("********************************")
        print("* Filtering with Initial Values*")
        print("********************************")
        self.filter_image()

    def filter_image(self, *widget_arg):
        """Call `image_filter` with widget args and kwargs

        Note: `display_filtered_image` is automatically called.
        """
        # `widget_arg` is passed by the active widget but is unused since all
        # filter arguments are pulled directly from attached the widgets.

        if self.image_filter is None:
            return
        arguments = [self._get_value(a) for a in self.arguments]
        kwargs = dict([(name, self._get_value(a))
                       for name, a in self.keyword_arguments.items()])
        filtered_filename = self.image_filter(*arguments, **kwargs)

        filtered_vol = nib.load(filtered_filename).get_data().transpose(2,1,0) #HARDCODED TRANSPOSE FOR NOW

        vlist = self.image_viewer.volumes
        imlist = self.image_viewer.images
        ai = self.image_viewer.active_image_index
        vlist[ai] = filtered_vol
        imlist[ai] = vlist[...,self.image_viewer.current_frames[ai]]
        self.display_filtered_image(imlist[ai])
        self.image_changed.emit(filtered)

if False:
	image = data.camera()
	image2 = data.coins()
	image3 = data.chelsea()
	viewer = ImageViewer([image, image2, image3], image_labels=['Image 1','Image 2',''])
	#viewer.set_image(data.chelsea(),i=1)	
elif False:
	image = data.camera()[:,:,None]*np.ones((1,1,16),dtype=np.uint8)
	image2 = data.coins()[:,:,None]*np.ones((1,1,8),dtype=np.uint8)
	image3 = data.moon()[:,:,None]*np.ones((1,1,4),dtype=np.uint8)
	image[...,-4:] = image3
	image3[...,-2:] = image[...,:2]

	viewer = ImageViewer([image, image2, image3], image_labels=['Image 1','Image 2',''])
elif False:
	import nibabel as nib
	nii = nib.load('/media/Data1/c-mind-data-copy/IRC04H_06M008/IRC04H_06M008_P_1_WIP_T1W_3D_IRCstandard32_SENSE_4_1.nii.gz')
	voldata = nii.get_data().astype(np.float64)
	voldata = voldata/voldata.max()
	from skimage import img_as_float
	voldata = img_as_float(voldata)
	voldata = voldata[:,::-1,::-1]
	viewer = ImageViewer(voldata.transpose([2,1,0]), image_labels=['Axial','Coronal','Sagittal'], ortho_viewer=True)
elif True:
    from skimage.filter.rank import median
    from skimage.morphology import disk

    from skimage.viewer.widgets import OKCancelButtons, SaveButtons
    def median_filter(image, radius):
        return median(image, selem=disk(radius))

    image = data.coins()
    viewer = ImageViewer([image, image.copy()], image_labels=['Original', 'Filtered'], active_image_index=1)

    plugin = Plugin(image_filter=median_filter)
    plugin += Slider('radius', 2, 10, value_type='int')
    plugin += SaveButtons()
    plugin += OKCancelButtons()
    
    viewer += plugin
    viewer.show()
elif False:
    from skimage.viewer.widgets import OKCancelButtons, SaveButtons, CheckBox

    import nibabel as nib
    nii_file = '/media/Data1/c-mind-data-copy/UCLA_03F003/UCLA_03F003_P_1_T1W_3D_MPRAGE_2_1.nii.gz'
    input_nii = nib.load(nii_file)
    voldata = input_nii.get_data().astype(np.float64)
    voldata = voldata/voldata.max()
    from skimage import img_as_float
    voldata = img_as_float(voldata.transpose(2,1,0))

    viewer = ImageViewer([voldata, voldata.copy()], image_labels=['Original', 'Defaced'], ortho_viewer=False)

    from cmind.pipeline.cmind_deface import cmind_deface
    defacer = partial(cmind_deface, output_nii = nii_file.replace('.nii.gz','_defaced_GUI.nii.gz'))

#[('IRC04H_02F003', 'P', 1), ('UCLA_03F003', 'P', 1), ('UCLA_03M003', 'P', 1), ('UCLA_07F002', 'P', 1), ('UCLA_07F003', 'P', 1), ('UCLA_07M006', 'P', 1), ('UCLA_08F002', 'P', 1), ('UCLA_08F006', 'P', 1), ('UCLA_08F008', 'P', 1), ('UCLA_08F009', 'P', 1), ('UCLA_08F010', 'P', 1), ('UCLA_08M002', 'P', 1), ('UCLA_08M004', 'P', 1), ('UCLA_09F004', 'P', 1), ('UCLA_09F005', 'P', 1), ('UCLA_09F006', 'P', 1), ('UCLA_09F007', 'P', 1), ('UCLA_09F009', 'P', 1), ('UCLA_09F010', 'P', 1), ('UCLA_09F011', 'P', 1), ('UCLA_09M002', 'P', 1), ('UCLA_09M004', 'P', 1), ('UCLA_09M005', 'P', 1)]


    plugin = CustomPlugin(image_filter=defacer, first_argument_to_filter=nii_file) # doctest: +SKIP
    plugin += Slider('age_months', 0, 2000, value = 252, value_type='int')
    plugin += Slider('crop_thresh', 0.001, 1, value = 0.03, value_type='float')
    plugin += Slider('iso_xfm_mm', 1, 2, value = 1, value_type='int')
    plugin += CheckBox(name='bias_cor', value = False)
    plugin += CheckBox(name='keep_intermediate_files', value = False)
    plugin += CheckBox(name='no_crop', value = False)
    plugin += CheckBox(name='generate_figures', value = False)
    plugin += CheckBox(name='verbose', value = True)
    plugin += CheckBox(name='ForceUpdate', value = False)
    viewer += plugin
    viewer.show()

#viewer.image = image[...,0] #data.camera()
#viewer.set_image(image2[...,-1],i=2)


viewer.show()

"""
TODO: change on_move to on_release for slow denoising case

TODO: add wait cursor during processing as suggested here
http://stackoverflow.com/questions/8218900/how-can-i-change-the-cursor-shape-with-pyqt



down vote
 

ekhumoro's solution is correct. This solution is a modification for the sake of style. I used what ekhumor's did but used a python decorator.
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication, QCursor, QMainWidget

def waiting_effects(function):
    def new_function(self):
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        function(self)
        QApplication.restoreOverrideCursor()
    return new_function

I can just put the decorator on any method I would like the spinner to be active on.
class MyWigdet(QMainWidget):

    # ...

    @waiting_effects
    def doLengthyProcess(self):
        # do lengthy process
        pass
 
"""
