import sys
import numpy as np
from pathlib import Path

# add parent dir to sys.path
sys.path.append('..')
import deferred_acceptance as da


class PreferenceGeneratorInterface:
    def generate_prefs(self, N, C):
        """Generate dictionary giving true and approximate student and college preferences, as well as any other information."""
        pass


def generate_student_prefs(N, C, beta=0, gamma=0):
    student_epsilons = [np.random.logistic(size=C) for _ in range(N)]
    student_locs = [np.random.uniform() for _ in range(N)]
    college_locs = [np.random.uniform() for _ in range(C)]
    college_qualities = [np.random.uniform() for _ in range(C)]
    student_values = [[student_epsilons[i][c] - gamma*(student_locs[i] - college_locs[c])**2 + beta*college_qualities[c] for c in range(C)] for i in range(N)]
    student_prefs = da.prefs_from_values(student_values)
    return student_prefs


class MonocultureGenerator(PreferenceGeneratorInterface):
    def __init__(self, noise_function, beta=0, gamma=0, access_distribution='total', strategy='top'):
        self.noise_function = noise_function
        self.beta = beta
        self.gamma = gamma
        self.access_distribution = access_distribution
        self.strategy = strategy

    def generate_prefs(self, N, C):
        pref_dict = {}

        """Generate student preferences."""
        student_prefs = generate_student_prefs(N, C, self.beta, self.gamma)

        num_apps = []
        if self.access_distribution=='uniform':
            limited_student_prefs = []
            for prefs in student_prefs:
                k = np.random.randint(1, C+1)
                num_apps.append(k)
                if self.strategy == 'top':
                    limited_student_prefs.append(prefs[:k])
                elif self.strategy == 'random':
                    limited_student_prefs.append(list(np.array(prefs)[np.sort(np.random.choice(range(C), k, replace=False))]))
                else:
                    raise ValueError('Invalid strategy')
        elif self.access_distribution=='total':
            limited_student_prefs = student_prefs
            num_apps = [C for _ in range(N)]
        else:
            raise ValueError('Invalid access distribution')

        pref_dict['approx_student'] = limited_student_prefs
        pref_dict['true_student'] = student_prefs
        pref_dict['num_apps'] = num_apps

        """Generate college preferences."""
        values = np.sort(np.random.uniform(size=N))
        scores = values + np.array([self.noise_function() for _ in range(N)])
        college_values = [scores for _ in range(C)]
        pref_dict['true_college'] = da.prefs_from_values([values for _ in range(C)])
        pref_dict['approx_college'] = da.prefs_from_values(college_values)
        pref_dict['true_college_shared'] = True

        return pref_dict
    

class PolycultureGenerator(PreferenceGeneratorInterface):
    def __init__(self, noise_function, beta=0, gamma=0, access_distribution='total', strategy='top'):
        self.noise_function = noise_function
        self.beta = beta
        self.gamma = gamma
        self.access_distribution = access_distribution
        self.strategy = strategy

    def generate_prefs(self, N, C):
        pref_dict = {}

        """Generate student preferences."""
        student_prefs = generate_student_prefs(N, C, self.beta, self.gamma)

        num_apps = []
        if self.access_distribution=='uniform':
            limited_student_prefs = []
            for prefs in student_prefs:
                k = np.random.randint(1, C+1)
                num_apps.append(k)
                if self.strategy == 'top':
                    limited_student_prefs.append(prefs[:k])
                elif self.strategy == 'random':
                    limited_student_prefs.append(list(np.array(prefs)[np.sort(np.random.choice(range(C), k, replace=False))]))
                else:
                    raise ValueError('Invalid strategy')
        elif self.access_distribution=='total':
            limited_student_prefs = student_prefs
            num_apps = [C for _ in range(N)]
        else:
            raise ValueError('Invalid access distribution')

        pref_dict['approx_student'] = limited_student_prefs
        pref_dict['true_student'] = student_prefs
        pref_dict['num_apps'] = num_apps

        """Generate college preferences."""
        values = np.sort(np.random.uniform(size=N))
        college_values = [values + np.array([self.noise_function() for _ in range(N)]) for _ in range(C)]
        pref_dict['true_college'] = da.prefs_from_values([values for _ in range(C)])
        pref_dict['approx_college'] = da.prefs_from_values(college_values)
        pref_dict['true_college_shared'] = True

        return pref_dict


class Market:
    def __init__(self, N, C, college_caps, PreferenceGenerator):
        self.N = N
        self.C = C
        self.college_caps = college_caps

        pref_dict = PreferenceGenerator.generate_prefs(N, C)
        self.true_student_prefs, self.approx_student_prefs = pref_dict['true_student'], pref_dict['approx_student']
        self.true_college_prefs, self.approx_college_prefs = pref_dict['true_college'], pref_dict['approx_college']
        self.num_apps = pref_dict['num_apps']
        if pref_dict['true_college_shared'] == True:
            self.student_percentile = [i / (self.N - 1) for i in range(self.N)]

        """Get stable matching according to approximate preferences."""
        self.student_matches, self.college_matches = da.get_match(self.approx_student_prefs, self.approx_college_prefs, college_caps)

    def student_rank_of_matches(self):
        """Get rank of college matched to each student according to true preferences. 0 if student is unmatched."""
        ranks = [0 for _ in range(self.N)]
        for i in range(self.N):
            if self.student_matches[i] != -1:
                ranks[i] = self.true_student_prefs[i].index(self.student_matches[i]) + 1    
        return ranks
    
    def num_apps_submitted(self):
        return self.num_apps
    
    def student_welfare(self):
        """Get average rank of college matched to each student according to true preferences (among matched students)."""
        return np.sum(self.student_rank_of_matches())/np.sum([1 if self.student_matches[i] != -1 else 0 for i in range(self.N)])
    
    def college_percentile_of_matches(self):
        """Get percentile rank of students matched to each college according to true preferences. 0 if college is unmatched."""
        percentiles = [[0 for _ in range(self.college_caps[c])] for c in range(self.C)]
        for i in range(self.N):
            if self.student_matches[i] != -1:
                percentiles[self.student_matches[i]].insert(0, i/(self.N - 1)) # percentile rank (1 is best), note that students are sorted lowest value to highest
                percentiles[self.student_matches[i]].pop()
        return percentiles
    
    def college_welfare(self):
        """Get average percentile rank of students matched to a college."""
        ranks = []
        for i in range(self.N):
            if self.student_matches[i] != -1:
                ranks.append(1 - self.true_college_prefs[self.student_matches[i]].index(i)/(self.N - 1)) # percentile rank (1 is best)

        return np.sum(ranks)/np.sum(self.college_caps) # average percentile rank (if a college does not fill capacity, empty slots are viewed as 0-th percentile)