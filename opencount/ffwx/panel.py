'''
The Panel class from which all other panel structures inherit.
'''

import wx
try:
    import cPickle as pickle
except ImportError:
    import pickle


class Panel(wx.Panel):
    '''
    A wrapper that all visible panels should inherit from.
    '''

    class StepNotFinished(Exception):
        '''
        An exception produced by the .can_move_on method in the
        event that more work needs to be done. The message will
        be translated into a user-facing error message.
        '''

        def __init__(self, message):
            self.message = message

        def __repr__(self):
            return 'StepNotFinished({0})'.format(repr(self.message))

    class SkipToStep(Exception):
        '''
        An exception produced in the event that the current step
        is not necessary; it should specify the next step needed
        in the ballot-counting workflow.
        '''

        def __init__(self, message, target_step):
            self.message = message
            self.target_step = target_step

        def __repr__(self):
            return 'SkipToStep({0}, {1})'.format(
                repr(self.message),
                self.target_step)

    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

    def ensure_partitioned(self, project):
        if not project.is_grouped():
            if project.is_partitioned():
                project.use_partitions_as_grouping()
            else:
                raise Panel.SkipToStep(
                    'You must first run partitioning before this '
                    'step. OpenCount will take you there now.',
                    2)

    def start(self, project=None, projdir=None):
        '''
        Set up the correct state for the tab
        '''
        raise NotImplentedError()

    def stop(self):
        '''
        Leave the relevant tab
        '''
        raise NotImplentedError()

    def can_move_on(self):
        '''
        Returns a boolean indicating whether the tasks associated
        with this panel have been completed, as well as a message
        to be used in the case that something is left to do.
        '''
        if self.run_sanity_checks():
            return True
        else:
            raise Panel.StepNotFinished('[ATTENTION NEEDED]')

    def run_sanity_checks(self):
        '''
        Return True if everything is fine
        '''
        return True

    def load_session_with(self, fields=[]):
        '''
        Given a list of field names, fill in the locals of this
        object from the saved versions
        '''
        if not self.statefileP:
            return False
        try:
            with open(self.statefileP, 'rb') as f:
                state = pickle.load(f)
            for f in fields:
                self.__dict__[f] = state[f]
            return True
        except:
            return False

    def save_session_with(self, fields=[]):
        '''
        Given a list of field names, save all of those to a state file.
        '''
        if not self.statefileP:
            return False
        try:
            with open(self.statefileP, 'wb') as f:
                state = dict((f, self.__dict__[f]) for f in fields)
                state = pickle.save(state, f, pickle.HIGHEST_PROTOCOL)
            return True
        except:
            return False
