package lr

import (
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

type model struct {
	w         vec64
	b         float64
	threshold float64
}

// threshold = (0.1, 1.0)
func LogisticRegressor(threshold float64) *model {
	return &model{threshold: threshold}
}

type RegressionModel interface {
	LoadFromFile(path string) error
	LoadToFile(path string) error
	Fit(X []vec64, y vec64, lambda_ float64, randomState, epochs, printAfterEach, earlyStopping int)
	Predict(X ...vec64) vec64
}

func (m *model) LoadToFile(path string) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer file.Close()
	sb := strings.Builder{}
	for i := range m.w {
		sb.WriteString(strconv.FormatFloat(m.w[i], 'f', 32, 64))
		sb.WriteByte(',')
	}
	sb.WriteString(strconv.FormatFloat(m.b, 'f', 32, 64))
	_, err = file.WriteString(sb.String())
	return err
}

func (m *model) LoadFromFile(path string) error {
	file, err := os.OpenFile(path, os.O_RDONLY, 0600)
	if err != nil {
		return err
	}
	defer file.Close()
	body, err := io.ReadAll(file)
	if err != nil {
		return err
	}
	vec := Vec(strings.Split(string(body), ",")...)
	last := len(vec) - 1
	m.w = vec[:last]
	m.b = vec[last]
	return nil
}

// lambda_ = 0 for disable reg
func (m *model) Fit(X []vec64, y vec64, lambda_ float64, randomState, epochs, printAfterEach, earlyStopping int) {
	featuren := len(X[0])
	wInit := make(vec64, featuren)
	rd := rand.New(rand.NewSource(int64(randomState)))
	for i := range featuren {
		wInit[i] = rd.Float64()
	}
	bInit := rd.Float64()
	lrate := 0.05

	m.w, m.b = gradientDescent(X, y, wInit, bInit, lrate, lambda_, epochs, printAfterEach, earlyStopping)
}

func (m *model) Predict(X ...vec64) vec64 {
	rows := len(X)
	p := make(vec64, rows)
	for i := range rows {
		if sigmoid(Dot(X[i], m.w)+m.b) > m.threshold {
			p[i] = 1
			continue
		}
		p[i] = 0
	}
	return p
}

func ComputePredictMetrics(yhat, y vec64) (tp, tn, fp, fn float64) {
	for i := range y {
		if y[i] == 1 {
			if y[i] == yhat[i] {
				tp++
			} else {
				fn++
			}
		} else {
			if y[i] == yhat[i] {
				tn++
			} else {
				fp++
			}
		}
	}
	return
}

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func computeGradient(X []vec64, y vec64, w vec64, b float64) (vec64, float64) {
	samplesNum, featuresNum := len(y), len(X[0])
	dj_dw := make(vec64, featuresNum)
	dj_db := 0.0

	for i := range samplesNum {
		yhati := sigmoid(Dot(X[i], w) + b)
		err := yhati - y[i]
		for j := range featuresNum {
			dj_dw[j] += err * X[i][j]
		}
		dj_db += err
	}

	return dj_dw.Divide(float64(samplesNum)), dj_db / float64(samplesNum)
}

func computeGradientReg(X []vec64, y vec64, w vec64, b, lambda float64) (vec64, float64) {
	samplesNum, featuresNum := len(y), len(X[0])
	dj_dw, dj_db := computeGradient(X, y, w, b)

	if lambda != 0 {
		for j := range featuresNum {
			dj_dw[j] += lambda / float64(samplesNum) * w[j]
		}
	}

	return dj_dw, dj_db
}

func gradientDescent(X []vec64, y, wInit vec64, bInit, learningRate, lambda float64, numIters, checkEach, earlyStopping int) (vec64, float64) {
	w := make(vec64, len(wInit))
	copy(w, wInit)
	b := bInit
	q := queue{cap: earlyStopping}
	for i := range numIters {
		dj_dw, dj_db := computeGradientReg(X, y, w, b, lambda)

		w = Subtract(w, dj_dw.Dot(learningRate))
		b -= learningRate * dj_db

		cost := BinaryCrossEntropy(X, y, w, b, lambda)
		q.insert(w, b, cost)
		if (i+1)%checkEach == 0 {
			log.Printf("Iteration: %d, cost: %.4f\n", i+1, cost)
		}
		if q.len == q.cap && cost >= q.head.val - 1e-6 {
			log.Printf("early stopped after %d iterations, with best score: %.4f", i, q.best.val)
			return q.best.w, q.best.b
		}
	}

	return w, b
}

func BinaryCrossEntropy(X []vec64, y, w vec64, b, lambda float64) float64 {
	cost := 0.0
	samples := len(y)
	epsilon := 1e-15
	for i := range samples {
		yhati := sigmoid(Dot(X[i], w) + b)
		cost += -y[i]*math.Log(yhati+epsilon) - (1-y[i])*math.Log(1-yhati+epsilon)
	}
	if lambda != 0 {
		cost += lambda / (float64(samples) * 2) * w.Pow(2.0).Sum()
	}
	return cost / float64(samples)
}

type queue struct {
	head, best, end *node
	len, cap        int
}

type node struct {
	next *node
	w    vec64
	b    float64
	val  float64
}

func (q *queue) insert(w vec64, b, val float64) {
	if q.len == 0 {
		q.head = &node{w: w, b: b, val: val}
		q.best = q.head
		q.end = q.head
		q.len = 1
		return
	}
	if q.len == q.cap {
		q.delete()
	}
	q.end.next = &node{w: w, b: b, val: val}
	q.end = q.end.next
	if q.best.val > q.end.val {
		q.best = q.end
	}
	q.len++
}

func (q *queue) delete() {
	if q.best == q.head {
		p := q.head.next
		q.best = p
		for p != nil {
			if p.val < q.best.val {
				q.best = p
			}
			p = p.next
		}
	}
	q.head = q.head.next
	q.len--
}