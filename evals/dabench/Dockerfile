# The analyst's interpreter: DA-Verify's image plus the scientific stack DABench asks for.
#
# DEPARTURE from da-verify (pandas + numpy only). 19 questions are tagged Machine
# Learning and name sklearn outright ("Apply the linear regression algorithm from the
# sklearn library..."), and the Distribution / Correlation questions ask for t-tests,
# Shapiro-Wilk and Pearson p-values, which is scipy. Without them a model that knows
# exactly what to do fails on `ModuleNotFoundError`, and the scoreboard reads that as a
# model that cannot do data analysis. matplotlib is here only so a model that reaches
# for a plot out of habit does not crash the cell; the Agg backend needs no display.
#
# Pinned so a rerun a month from now computes the same p-values: a scipy default
# changing (e.g. `ttest_ind`'s `equal_var`) moves gold-matching answers.
FROM python:3.12-slim
RUN pip install --no-cache-dir \
    pandas==2.2.3 numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 \
    statsmodels==0.14.4 matplotlib==3.9.2
ENV MPLBACKEND=Agg
WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]
