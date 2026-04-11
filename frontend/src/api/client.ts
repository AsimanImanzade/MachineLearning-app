import axios from 'axios'

const api = axios.create({
  // Hardcoded fallback for production stability + '/api' suffix matching FastAPI routers
  baseURL: (import.meta.env.VITE_API_URL || 'https://machinelearning-app.onrender.com') + '/api',
  headers: { 'Content-Type': 'application/json' },
})

export default api

// ── Linear Regression ──────────────────────────────────────────────────────
export const linearTrain = (params: object) => api.post('/linear/train', params).then(r => r.data)
export const linearOverfit = (params: object) => api.post('/linear/overfit-sandbox', params).then(r => r.data)
export const linearRealDataset = (params: object) => api.post('/linear/real-dataset', params).then(r => r.data)

// ── Logistic Regression ────────────────────────────────────────────────────
export const logisticTrain = (params: object) => api.post('/logistic/train', params).then(r => r.data)

// ── KNN ────────────────────────────────────────────────────────────────────
export const knnTrain = (params: object) => api.post('/knn/train', params).then(r => r.data)
export const knnOverfitCurve = (params: object) => api.post('/knn/overfit-curve', params).then(r => r.data)

// ── Decision Tree ──────────────────────────────────────────────────────────
export const dtTrain = (params: object) => api.post('/decision-tree/train', params).then(r => r.data)
export const dtOverfitCurve = (params: object) => api.post('/decision-tree/overfit-curve', params).then(r => r.data)

// ── Random Forest ──────────────────────────────────────────────────────────
export const rfRegression = (params: object) => api.post('/random-forest/regression', params).then(r => r.data)
export const rfClassification = (params: object) => api.post('/random-forest/classification', params).then(r => r.data)
export const rfCrossValidate = (params: object) => api.post('/random-forest/cross-validate', params).then(r => r.data)
export const rfHousingDataset = () => api.get('/random-forest/housing-dataset').then(r => r.data)
