/**
 * Created by joonkukang on 2014. 1. 19..
 */
let ml = require('../lib/machine_learning');

let data = [
    [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]
];

let result = [23, 12, 23, 23, 45, 70, 123, 73, 146, 158, 64];

let knn = new ml.KNN({
    data: data,
    result: result
});

let y = knn.predict({
    x: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    k: 3,
    weightf: {type: 'gaussian', sigma: 10.0},
    distance: {type: 'euclidean'}
});

console.log(`y value predict: ${y}`);

let data_film = [
    [3, 104],
    [4, 100],
    [1, 81],
    [101, 10],
    [99, 3],
    [98, 5]
];

// let result_film = ['喜剧片', '喜剧片', '喜剧片', '动作片', '动作片', '动作片'];
let result_film = [0, 0, 0, 1, 1, 1];

let knn_film = new ml.KNN({
    data: data_film,
    result: result_film
});

let category = knn_film.predict({
    x: [18, 90],
    k: 3,
    weightf: {type: 'gaussian', sigma: 10.0},
    distance: {type: 'euclidean'}
});

console.log(`movie type predict: ${category}`)