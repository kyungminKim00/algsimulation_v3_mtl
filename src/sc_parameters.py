"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import, division, print_function

from header.index_forecasting.RUNHEADER import (
    domain_search_parameter,
    forward_map,
    mkidx_mkname,
    mkname_dataset,
    mkname_mkidx,
)


class ScriptParameters:
    def __init__(self, domain, m_args, job_id_int=None, search_parameter=None):

        self.process_id = None
        self.ref_pid = None
        self.m_target_name = domain.split("_")[0]
        self.forward_ndx = int(domain.split("_")[1])
        self.forward_idx = forward_map[self.forward_ndx]
        self.m_target_index = mkname_mkidx[self.m_target_name]
        self.dataset_version = mkname_dataset[self.m_target_name]
        self.m_args = m_args

        if search_parameter is None:
            self.search_parameter = domain_search_parameter[domain]
        else:
            self.search_parameter = search_parameter

        if job_id_int is not None:
            self.job_id_int = int(job_id_int)
            ref_pid = self.dataset_version[1:] + str(self.forward_idx)
            self.process_id = int(ref_pid + str(self.job_id_int))
            self.ref_pid = int(ref_pid)

    def update_args(self):
        self.m_args.m_target_index = self.m_target_index
        self.m_args.dataset_version = self.dataset_version
        self.m_args.forward_ndx = self.forward_ndx
        self.m_args.process_id = self.process_id
        self.m_args.ref_pid = self.ref_pid
        self.m_args.search_parameter = self.search_parameter
        return self.m_args
