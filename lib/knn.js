/**
 * Created by joonkukang on 2014. 1. 16..
 */
let math = require('./utils').math;
let KNN = module.exports = function (options) {
    let self = this;
    self.data = options['data'];
    self.result = options['result'];
}

KNN.prototype.predict = function (options) {
    let self = this;
    let x = options['x'];
    let k = options['k'] || 3;
    let weightf = getWeightedFunction(options['weightf']);
    let distance = getDistanceFunction(options['distance']);
    let distanceList = [];
    let i;
    for (i = 0; i < self.data.length; i++)
        distanceList.push([distance(x, self.data[i]), i]);
    distanceList.sort(function (a, b) {
        return a[0] - b[0];
    });
    let avg = 0.0;
    let totalWeight = 0, weight;
    for (i = 0; i < k; i++) {
        let dist = distanceList[i][0];
        let idx = distanceList[i][1];
        weight = weightf(dist);
        avg += weight * self.result[idx];
        totalWeight += weight;
    }

    avg /= totalWeight;
    return avg;
};

function getWeightedFunction(options) {
    if (typeof options === 'undefined') {
        return function (x) {
            let sigma = 10.0;
            return Math.exp(-1. * x * x / (2 * sigma * sigma));
        }
    } else if (typeof options === 'function') {
        return options;
    } else if (options['type'] === 'gaussian') {
        return function (x) {
            let sigma = options['sigma'];
            return Math.exp(-1. * x * x / (2 * sigma * sigma));
        }
    } else if (options['type'] === 'none') {
        return function (dist) {
            return 1.0;
        }
    }
}

function getDistanceFunction(options) {
    if (typeof options === 'undefined') {
        return math.euclidean;
    } else if (typeof options === 'function') {
        return options;
    } else if (options['type'] === 'euclidean') {
        return math.euclidean;
    } else if (options['type'] === 'pearson') {
        return math.pearson;
    }
}
