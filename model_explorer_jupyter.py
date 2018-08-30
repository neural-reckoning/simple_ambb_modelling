import ipywidgets as ipw
from IPython.display import display, clear_output
from collections import OrderedDict
import numpy
import os
import pickle

__all__ = ['no_continuous_update',
           'model_explorer',
           'VariableSelector',
           'meshed_arguments',
           'save_fig_widget',
           'brian2_progress_reporter',
           'load_save_parameters_widget',
           ]

def no_continuous_update(i):
    '''
    Switches off continous updates for ipywidgets.interactive
    
    Usage: no_continuous_update(interactive(...))
    '''
    for w in i.children:
        w.continuous_update = False
    return i


def save_fig_widget(callback, default_filename="figure.pdf"):
    '''
    Returns a save figure button and filename text box
    
    When clicked, calls the function callback with the filename as argument
    '''
    savefig_button = ipw.Button(description="Save figure")
    widget_filename = ipw.Text(value=default_filename)
    def save_fig_callback(*args):
        callback(widget_filename.value)
    savefig_button.on_click(save_fig_callback)
    return ipw.HBox(children=[savefig_button, widget_filename])
           
    
def brian2_progress_reporter():
    '''
    Returns a widget and callback function to be used with Brian 2
    '''
    progress_slider = ipw.FloatProgress(description="Simulation progress", min=0, max=1)
    progress_slider.layout.width = '100%'
    def update_progress(elapsed, complete, t_start, duration):
        progress_slider.value = complete
        if complete==1:
            progress_slider.bar_style = 'success'
        else:
            progress_slider.bar_style = ''    
    return progress_slider, update_progress

    
def model_explorer(models):
    '''
    Returns a Model Explorer UI
    
    models
        A list of items of the forms either (name, func), (name, func, options),
        or (name, func, options, additional_controls): where options are shown on
        the model type and optiosn tab, and additional controls are shown on the
        results tab.
    name
        The name of the model
    func
        Function which takes the options as arguments (if there are any), and
        returns an interactive widget (provided by the ``ipywidgets.interactive``
        function). This function gets called when the UI tab is changed from
        model type and options to parameters and results.
    options
        Dictionary of option names / widget pairs to give additional options
        for each model type. If the widget has a ``.value`` attribute, this
        value will be passed to func.
        
    Returned modex widget has some additional attributes:
    
    modex.tabs (Tab)
        The overall Tab widget (set the index of this to jump immediately to results).
    model.widget_model_type (RadioButtons)
        The model type widget (set this before changing the tab index to choose a particular model type)
        
    See examples in documentation for more details.
    '''
    all_model_names = []
    all_model_funcs = {}
    all_model_options = {}
    all_model_additional_controls = {}
    for model in models:
        model_options = {}
        model_additional_controls = []
        if len(model)==2:
            model_name, model_func = model
        elif len(model)==3:
            model_name, model_func, model_options = model
        elif len(model)==4:
            model_name, model_func, model_options, model_additional_controls = model
        else:
            raise ValueError("Each model must have 2, 3 or 4 items")
        all_model_names.append(model_name)
        all_model_funcs[model_name] = model_func
        all_model_options[model_name] = model_options
        all_model_additional_controls[model_name] = model_additional_controls

    ################# PANEL FOR CHOOSING MODEL TYPE AND OPTIONS
    # Update the options for the GUI if the model type changes
    def change_model_type(change):
        model = change['new']
        options = all_model_options[model].values()
        if len(options): # add a header for the options
            options = [ipw.HTML('''
                <p style="margin-bottom: 10px; border-bottom: 1px solid black;">
                    Options for {model}
                </p>'''.format(model=model))]+options
        widget_options_container.children = options

    widget_model_type = ipw.RadioButtons(description="Model type", options=all_model_names)
    widget_options_container = ipw.VBox()
    change_model_type(dict(name='value', new=all_model_names[0]))
    widget_model_type.observe(change_model_type, names='value')
    widget_model_type_and_options = ipw.VBox(children=[widget_model_type, widget_options_container])

    ################# PANEL FOR SLIDERS AND MODEL DISPLAY
    def html_header(text):
        return ipw.HTML('''
            <p style="margin-bottom: 10px; margin-top: 10px; border-bottom: 1px solid black;">
                {text}
            </p>'''.format(text=text))
    widget_interact_container = ipw.VBox()
    additional_controls_container = ipw.VBox()
    params_widgets = [html_header('Parameters'), widget_interact_container, additional_controls_container]
    widget_parameters_container = ipw.VBox(children=params_widgets)

    ################# OVERALL TABS FOR GUI
    def change_tab(change):
        if change['new']==1:
            model = widget_model_type.value
            option_widgets = all_model_options[model]
            options = {}
            for k, v in option_widgets.items():
                if hasattr(v, 'value'):
                    options[k] = v.value
            interactive_widget = all_model_funcs[model](**options)
            widget_interact_container.children = [interactive_widget]
            additional = all_model_additional_controls[model]
            if len(additional):
                additional_controls_container.children = [html_header('Additional controls')]+additional
            # hack to make it run the model immediately
            interactive_widget._display_callbacks(widget_interact_container)

    tab_container = ipw.Tab(children=[widget_model_type_and_options, widget_parameters_container])
    tab_container.set_title(0, "Model type and options")
    tab_container.set_title(1, "Parameters and results")
    tab_container.observe(change_tab, names='selected_index')
    container = ipw.VBox(children=[tab_container])
    
    container.tabs = tab_container
    container.widget_model_type = widget_model_type
    container.additional_controls_container = additional_controls_container

    return container


class VariableSelector(object):
    '''
    Utility for creating variable selection
    
    Arguments:
    
    vars
        A list or dictionary of variables to choose from. If a dictionary, then the keys will
        be the names of the variables, and the values the descriptions. Make sure to use
        OrderedDict if you want to present the variables in a given order.
    choices
        A list or dictionary (as for vars) of the choices offered to the user, e.g. could be
        ``choices=["Horizontal axis", "Vertical axis"]``.
    title
        A string labelling the widgets (or None).
    initial (optional)
        Dictionary mapping initial choice selection (choice names as keys, var names as values)
        
    Attributes:
    
    vs.widgets_horizontal
        List of widgets arranged horizontally (looks better)
    vs.widgets_vertical
        List of widgets arranged vertically (to be used if making a large number of choices)
    vs.selected
        Selected variables as a tuple in the order of the choices
    vs.unselected
        Unselected variables as a tuple in the order of the variables
        
    Methods:
        
    vs.delete_selected_from(d)
        Removes items from dictionary d if their key is in selected variables
    '''
    def __init__(self, vars, choices, title="Variable selection", initial=None):
        if not isinstance(vars, dict):
            vars = OrderedDict((k, k) for k in vars)
        if not isinstance(choices, dict):
            choices = OrderedDict((k, k) for k in choices)
        self.vars = vars
        self.choices = choices
        # inverted version of vars, to make descriptions to names
        self.vardesc2var = dict((v, k) for k, v in self.vars.items())
        self._widgets = OrderedDict()
        # setup initial selection
        self.selection = dict()
        for (choicename, choicedesc), var in zip(choices.items(), vars.keys()[:len(choices)]):
            self.selection[choicename] = var
        if initial is not None:
            self.selection = initial
        selected = set(self.selection.values())
        leftovervars = [vardesc for var, vardesc in self.vars.items() if var not in selected]
        # this function is called when one oi the variables is changed, to
        # update the choices for the other variables. If you've changed one
        # variable to a value already taken by another choice, then that
        # choice will be swapped with the one that the user clicked.
        def change_var(change):
            # don't do anything if we're currently in a change_var call
            # to stop an infinite regression
            if hasattr(change_var, 'locked') and change_var.locked:
                return
            change_var.locked = True
            old = change['old']
            new = change['new']
            owner = change['owner']
            self.selection[owner.name] = self.vardesc2var[new]
            for widget in self._widgets.values():
                if widget is owner:
                    continue
                if widget.value==new:
                    widget.value = old
                    self.selection[widget.name] = self.vardesc2var[old]
            change_var.locked = False # unlock to re-enable the function
        # create the widgets
        for choicename, choicedesc in choices.items():
            var = self.selection[choicename]
            options = self.vars.values()
            self._widgets[choicename] = w = ipw.RadioButtons(description=choicedesc,
                                                             options=options, value=self.vars[var])
            w.name = choicename
            w.layout.width = '100%'
            w.style = {'description_width': '30%'}
            # this calls the change_var function if the value of any of the selectors changes
            w.observe(change_var, names='value')
        self.widgets = self._widgets.values()
        if title is not None:
            # Use an HTML header to style it nicely
            header = [ipw.HTML('''
                <p style="margin-bottom: 10px; margin-top: 10px; border-bottom: 1px solid black;">
                    {title}
                </p>'''.format(title=title))]
        else:
            header = []
        # Create horizontal and vertical layouts
        self.widgets_vertical = ipw.VBox(header+self.widgets)
        hspace = ipw.HTML('''<div style="width: 20px;"/>''')
        hwidgets = [self.widgets[0]]
        for w in self.widgets[1:]:
            hwidgets.append(hspace)
            hwidgets.append(w)
        self.widgets_horizontal = ipw.VBox(header+[ipw.HBox(hwidgets)])
    
    @property
    def selected(self):
        '''
        Selected variables as a tuple in the order of the choices
        '''
        return tuple(self.selection[choicename] for choicename in self.choices.keys())
    
    @property
    def unselected(self):
        '''
        Unselected variables as a tuple in the order of the variables
        '''
        sel = set(self.selected)
        return tuple(k for k in self.vars.keys() if k not in sel)
    
    def delete_selected_from(self, d):
        '''
        Removes items from dictionary d if their key is in selected variables
        '''
        d = d.copy()
        for k in self.selected:
            if k in d:
                del d[k]
        return d
    
    def take_selected_from(self, d):
        '''
        Returns items from dictionary d if their key is in selected variables
        '''
        d = d.copy()
        for k in self.deleted:
            if k not in d:
                del d[k]
        return d
    
    def merge_selected(self, selected, unselected):
        '''
        Returns dictionary with keys in vars using values from dictionaries selected or unselected.
        '''
        d = OrderedDict()
        for k in self.vars.keys():
            if k in self.selected:
                d[k] = selected[k]
            else:
                d[k] = unselected[k]
        return d
    
    
def meshed_arguments(meshvars, fixedvars, ranges):
    '''
    Returns a dictionary of arguments representing all combinations of certain variables
    
    meshvars : list
        The names of the variables that should be meshed, in order
    fixedvars : dict
        Mapping names of fixed variables to their values
    ranges : dict
        Mapping names of meshed variables to their array of values
        
    Returns a dictionary mapping names of all variables (meshed and fixed) to their values.
    For fixed values, this will be just their value, and for meshed variables it will be
    a len(meshvars)-dimensional array of values.
    
    For example:
    
        kwds = meshed_arguments(['x', 'y'],
                                {'z': 0.0},
                                {'x': linspace(0, 1, 10), 'y': linspace(1, 2, 5)})
    
    Here, ``kwds['z']`` will be just 0.0, whereas:
    
        kwds['x'], kwds['y'] = meshgrid(linspace(0, 1, 10), linspace(1, 2, 5))
    '''
    kwds = fixedvars.copy()
    for i, c in enumerate(numpy.meshgrid(*[ranges[var] for var in meshvars])):
        kwds[meshvars[i]] = c
    return kwds


def load_save_parameters_widget(widgets, filename):
    '''
    Returns a box of widgets that manage saving/loading values from a given set of parameters.
    '''
    if os.path.exists(filename):
        param_sets = pickle.load(open(filename, 'rb'))
    else:
        param_sets = {}
    def getparams():
        params = {}
        for name, widget in widgets.items():
            params[name] = widget.value
        return params
    def setparams(params):
        for name, widget in widgets.items():
            widget.value = params[name]
    def saveparams():
        pickle.dump(param_sets, open(filename, 'wb'), -1)
        dropdown.options = sorted(param_sets.keys())
    def clicked_save(*args):
        param_sets[textbox.value] = getparams()
        saveparams()
        dropdown.value = textbox.value
    def clicked_load(*args):
        textbox.value = dropdown.value
        setparams(param_sets[dropdown.value])
    def clicked_delete(*args):
        del param_sets[dropdown.value]
        saveparams()
    def clicked_delete_all(*args):
        param_sets.clear()
        saveparams()
    description = ipw.Label(value="Parameter set:")
    dropdown = ipw.Dropdown(options=sorted(param_sets.keys()))
    save_params_button = ipw.Button(description='Save')
    save_params_button.on_click(clicked_save)
    textbox = ipw.Text()
    load_params_button = ipw.Button(description='Load')
    load_params_button.on_click(clicked_load)
    delete_params_button = ipw.Button(description='Delete')
    delete_params_button.on_click(clicked_delete)
    delete_all_params_button = ipw.Button(description='Delete all')
    delete_all_params_button.on_click(clicked_delete_all)
    save = ipw.HBox(children=[dropdown, load_params_button, delete_params_button, delete_all_params_button])
    load = ipw.HBox(children=[textbox, save_params_button])
    loadsave = ipw.VBox(children=[save, load])
    return ipw.HBox(children=[description, loadsave])
