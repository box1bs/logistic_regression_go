package lr

import (
	"encoding/csv"
	"math/rand"
	"os"
)

func UploadDataset(path string, hasNamedCols bool) ([]vec64, vec64, error) {
	file, err := os.OpenFile(path, os.O_RDONLY, 0600)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()
	r := csv.NewReader(file)
	records, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if hasNamedCols {
		records = records[1:]
	}
	X := make([]vec64, 0, 64)
	scores := make([]string, 0, 64)
	shape := len(records[0])
	for i := range records {
		X = append(X, Vec(records[i][:shape-1]...))
		scores = append(scores, records[i][shape-1])
	}
	y := Vec(scores...)
	return X, y, nil
}

func SplitSample(X []vec64, y vec64, scale float64) ([]vec64, []vec64, vec64, vec64) {
	sampleNum := len(X)
	scaled := int(float64(sampleNum) * scale)
	r := rand.New(rand.NewSource(42))
	generated := r.Perm(sampleNum)
	trainIdx := generated[:scaled]
	testIdx := generated[scaled:]
	trainX, testX, trainY, testY := make([]vec64, scaled), make([]vec64, sampleNum-scaled), make(vec64, scaled), make(vec64, sampleNum-scaled)
	for i := range scaled {
		trainX[i], trainY[i] = X[trainIdx[i]], y[trainIdx[i]]
		if i < sampleNum-scaled {
			testX[i], testY[i] = X[testIdx[i]], y[testIdx[i]]
		}
	}
	return trainX, testX, trainY, testY
}