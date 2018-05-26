let points = [] 
let predictedLine = []
let model = undefined

function setup(){
    buildModel()
    let cnvs = createCanvas(500, 500)
    cnvs.parent('container')
}

function drawLine(){
    if (!predictedLine.length)
        return
    let x1 = map(predictedLine[0].x, 0, 1, 0, width)
    let y1 = map(predictedLine[0].y, 0, 1, 0, height)
    let x2 = map(predictedLine[1].x, 0, 1, 0, width)
    let y2 = map(predictedLine[1].y, 0, 1, 0, height)
    line(x1, y1, x2, y2)
    // console.log(predictedLine[0].x,predictedLine[0].y,predictedLine[1].x,predictedLine[1].y)
}

function draw(){
    background(235)
    fill(0)
    points.forEach(p => {
        let x = map(p.x, 0, 1, 0, width)
        let y = map(p.y, 0, 1, 0, height)
        ellipse(x, y, 5, 5)
    })
    drawLine()
}

function mousePressed() {
    if (mouseX > width || mouseY > height || mouseX < 0 || mouseY < 0)
        return
    points.push(createVector(map(mouseX, 0, width, 0, 1), map(mouseY, 0, height, 0, 1)))
}

function buildModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.sgd(0.25)
    });
}

function trainModel(episodes) {
    return new Promise((resolve, reject) => {
        const features = tf.tensor1d(points.map(p => p.x))
        const targets = tf.tensor1d(points.map(p => p.y))
        for (let episode = 0; episode < episodes; episode++) {
            model.fit(features, targets, {
                epochs: 1
            })
            .then(h => {
                console.log("Loss after Epoch " +  episode + " : " + h.history.loss[0]);
                resolve()
                features.dispose()
                targets.dispose()
            })
        }
    })
}

setInterval(()=>{
    if(points.length > 3){
        trainModel(1)
        .then(()=>{
            tf.tidy(()=>{
                let x1 = 0
                let x2 = 1
                let [y1, y2] = model.predict(tf.tensor2d([x1, x2], [2, 1])).dataSync()
                predictedLine[0] = createVector(x1, y1)
                predictedLine[1] = createVector(x2, y2)
                console.log(tf.memory().numTensors)

            })
        })
    }
}, 25)