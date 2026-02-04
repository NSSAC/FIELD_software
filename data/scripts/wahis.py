DESC = '''
Converting the WAHIS pdf.

AA
'''

from kbdata import loader
from bs4 import BeautifulSoup
import click
from datetime import datetime
from glob import glob
import numpy as np
import pandas as pd
from pdb import set_trace
import re

def is_datetime_format(s, fmt="%Y/%m/%d"):
    try:
        datetime.strptime(s, fmt)
        return True
    except ValueError:
        return False

@click.group()
def cli():
    pass

def read_reports():
    states = loader.load('usa_states')
    i = 0
    recs = []
    pars_list = []
    report_num = sorted([int(x[16:-5]) for x in glob('temp_wahis/temp-*.html')])
    for rnum in report_num:
        print(rnum)
        report = f'temp_wahis/temp-{rnum}.html'
        with open(report, 'r', encoding='utf-8') as f:
            html = f.read()
            if 'bovine' not in html:
                continue
        soup = BeautifulSoup(html, 'html.parser')

        # collect all <p>
        pars_list.append(soup.find_all('p'))

    pars = [x for xx in pars_list for x in xx]
    
    # separate reports
    inc_start = []
    for i,par in enumerate(pars):
        if 'OB_' in par.get_text():
            inc_start.append(i)

    states_list = states.state.to_list()

    for i in range(len(inc_start)):
        try:
            inc = pars[inc_start[i]:inc_start[i+1]]
        except:
            inc = pars[inc_start[i]:]    # for the last one

        bovine_flag = False
        for par in inc:
            if 'bovine' in par.get_text():
                bovine_flag = True
                break
                
        if not bovine_flag: 
            continue

        rd = {}
        rd['outbreak_reference'] = re.search(r'OB_([0-9]+)', 
                                             inc[0].get_text()).group(1)

        start_date_flag = False
        for j,par in enumerate(inc):
            if not start_date_flag and is_datetime_format(par.get_text()):
                rd['start_date'] = inc[j].get_text()
                rd['end_date'] = inc[j+1].get_text()
                start_date_flag = True
            if start_date_flag and par.get_text().lower() in states_list:
                rd['state'] = inc[j].get_text().lower()
                rd['county'] = inc[j+1].get_text().lower()
                break

        recs.append(rd)

    df = pd.DataFrame.from_records(recs)
    df.start_date = pd.to_datetime(df.start_date)
    df = df[df.start_date.dt.year>=2023]

    # fips
    counties = loader.load('usa_counties')
    df = df.merge(counties[['state', 'county', 'county_code']], on=['state', 'county'])

    df.end_date = pd.to_datetime(df.end_date, errors='coerce')

    
    df.to_csv('dairy.csv', index=False)

if __name__ == '__main__':
    read_reports()
