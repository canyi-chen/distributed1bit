function beta_positive = force_first_positive(beta)
%FORCE_FIRST_POSITIVE Make sure the first element of beta ot be positive 
%
% Positional parameters:
%   beta    vector
% Return values:
%   beta_positive beta*sign(beta(1))
% Examples:
%   beta = -1*ones(10,1);
%   beta_positive = force_first_positive(beta)
%
    idx = find(beta); % indices of non-zero elements of betaT
    if ~isempty(beta)
        beta_positive = beta*beta(idx(1)); % enforce the first non-zero element is positive
    else
        beta_positive = beta;
    end
end

