class FinderState(object):
    def __init__(self, path_step, action_chosen=None,
                 history_state=None, action_prob=None,
                 tape=None,
                 prev_state=None, next_state=None):
        if prev_state is None and next_state is not None:
            self.path = (path_step,)
            self.entities = (path_step,)
            self.action_probs = ()
            self.action_chosen = ()
            self.tapes = ()
        if prev_state is not None:
            self.path = prev_state.path + path_step
            self.entities = prev_state.entities + (path_step[1],)
            self.action_probs = prev_state.action_probs + (action_prob,)
            self.action_chosen = prev_state.action_chosen + (action_chosen,)
            self.tapes = prev_state.tapes + (tape,)
        if next_state is not None:
            self.path = path_step + prev_state.path
            self.entities = (path_step[1],) + prev_state.entities
            self.action_probs = (action_prob,) + prev_state.action_probs
            self.action_chosen = (action_chosen,) + prev_state.action_chosen
            self.tapes = (tape,) + prev_state.tapes
        self.history_state = history_state

    def __str__(self):
        return " -> ".join(self.path)

    __repr__ = __str__
