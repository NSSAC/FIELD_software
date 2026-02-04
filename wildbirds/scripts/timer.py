DESC='''
Helper functions

AA
'''

from time import time


class timer:
    def __init__(self, unit='minutes', print_func=print, quiet=False):
        self.unit = unit
        self.start = time()
        self.current = self.start
        self.diff = None
        self.print_func = print_func
        self.time_segments = {}
        self.quiet = quiet

    def update(self, segment=None):
        curr = time()
        self.diff = curr - self.current
        self.current = curr
        if segment is not None:
            self.time_segments[segment] = self.diff

    def display(self, t, segment=''):
        if not segment:
            segment = 'Time: '
        if self.quiet:
            return
        if self.unit == 'minutes':
            self.print_func(f'{segment} {t/60: .0f}')
        elif self.unit == 'seconds':
            self.print_func(f'{segment} {t: .0f}')
        else:
            self.print_func(f'Unsupported time unit {self.unit}.')

    def update_and_display(self, segment=None):
        self.update(segment=segment)
        self.display(self.diff, segment=segment)

    def display_total_time(self):
        self.display(time()-self.start, 'Total time:')

    def summary(self):
        for x,y in self.time_segments.items():
            self.display(y, segment=x)
        self.display(time()-self.start, 'Total time:')
