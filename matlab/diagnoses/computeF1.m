function [f1,precision,recall] = computeF1(esupp, supp)
tp = length(intersect(esupp,supp));
fp = length(setdiff(esupp,supp));
fn = length(setdiff(supp,esupp));
precision = tp/(tp+fp);
recall = tp/(tp + fn);
f1 = 2*(precision*recall)/(precision+recall);
if precision == 0 | recall == 0
    f1 = 0;
end
end