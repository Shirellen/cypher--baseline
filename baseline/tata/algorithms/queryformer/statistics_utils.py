import pandas as pd
import numpy as np
import csv, scipy

columns = {'aka_name' : [
    'id',
    'person_id',
    'name',
    'imdb_index',
    'name_pcode_cf',
    'name_pcode_nf',
    'surname_pcode',
    'md5sum'
],
'aka_title' : [
    'id',
    'movie_id',
    'title',
    'imdb_index',
    'kind_id',
    'production_year',
    'phonetic_code',
    'episode_of_id',
    'season_nr',
    'episode_nr',
    'note',
    'md5sum'
],
'cast_info' : [
    'id',
    'person_id',
    'movie_id',
    'person_role_id',
    'note',
    'nr_order',
    'role_id'
],
'char_name': [
    'id',
    'name',
    'imdb_index',
    'imdb_id',
    'name_pcode_nf',
    'surname_pcode',
    'md5sum'
],
'comp_cast_type' : [
    'id',
    'kind'
],
'company_name' : [
    'id',
    'name',
    'country_code',
    'imdb_id',
    'name_pcode_nf',
    'name_pcode_sf',
    'md5sum'
],
'company_type' : [
    'id',
    'kind'
],
'complete_cast': [
    'id',
    'movie_id',
    'subject_id',
    'status_id'
],
'info_type' : [
    'id',
    'info'
],
'keyword': [
    'id',
    'keyword',
    'phonetic_code'
],
'kind_type' : [
    'id',
    'kind'
],
'link_type' : [
    'id',
    'link'
],
'movie_companies' : [
    'id',
    'movie_id',
    'company_id',
    'company_type_id',
    'note'
],
'movie_info_idx' : [
    'id',
    'movie_id',
    'info_type_id',
    'info',
    'note'
],
'movie_keyword' : [
    'id',
    'movie_id',
    'keyword_id'
],
'movie_link' : [
    'id',
    'movie_id',
    'linked_movie_id',
    'link_type_id'
],
'name': [
    'id',
    'name',
    'imdb_index',
    'imdb_id',
    'gender',
    'name_pcode_cf',
    'name_pcode_nf',
    'surname_pcode',
    'md5sum'
],
'role_type' : [
    'id',
    'role'
],
'title': [
    'id',
    'title',
    'imdb_index',
    'kind_id',
    'production_year',
    'imdb_id',
    'phonetic_code',
    'episode_of_id',
    'season_nr',
    'episode_nr',
    'series_years',
    'md5sum'
],
'movie_info': [
    'id',
    'movie_id',
    'info_type_id',
    'info',
    'note'
],
'person_info': [
    'id',
    'person_id',
    'info_type_id',
    'info',
    'note'
]}

hist_file = pd.read_csv('../Transformer_Query/data/histogram/histogram_string.csv')
for i in range(len(hist_file)):
    freq = hist_file['freq'][i]
    freq_np = np.frombuffer(bytes.fromhex(freq), dtype=np.float)
    hist_file['freq'][i] = freq_np

table_column = []
for i in range(len(hist_file)):
    table = hist_file['table'][i]
    col = hist_file['column'][i]
    table_alias = ''.join([tok[0] for tok in table.split('_')])
    if table == 'movie_info_idx': table_alias = 'mi_idx'
    combine = '.'.join([table_alias,col])
    table_column.append(combine)
hist_file['table_column'] = table_column

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

hist_new = hist_file.copy()
nbin = 50
for i in range(9):
    table = hist_new['table'][i]
    column = hist_new['column'][i]
    
    # read database
    data = pd.read_csv('test_files_open_source/imdb_data_csv/{}.csv'.format(table))
    data.columns = columns[table]
    
    col_data = np.array(data[column])
    data_filter = col_data[np.logical_not(np.isnan(col_data))].astype('int')
    
    length = len(data_filter)
    
    freq = np.bincount(data_filter) / (1.0 * length)
    bins = histedges_equalN(data_filter, nbin).astype('int')
    
    hist_new['freq'][i] = freq
    hist_new['bins'][i] = bins