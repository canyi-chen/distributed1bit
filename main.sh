#!/usr/bin/env bash

(matlab -nodisplay -nodesktop -r "if isunix cd matlab/cvx-a64; cvx_setup; end; if ismac cd matlab/cvx-maci64; cvx_setup; end; if ispc cd matlab/cvx-w64; cvx_setup; end; cd ../../; addpath(genpath('.')); run_all")  > output.log 2> errors.log
