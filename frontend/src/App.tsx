import { BrowserRouter, Routes, Route } from 'react-router-dom'
import PageLayout from './components/layout/PageLayout'
import Home from './pages/Home'
import StandardLR from './pages/linear-regression/StandardLR'
import LassoLR from './pages/linear-regression/LassoLR'
import RidgeLR from './pages/linear-regression/RidgeLR'
import ElasticNetLR from './pages/linear-regression/ElasticNetLR'
import LogisticReg from './pages/logistic-regression/LogisticReg'
import KNNClassification from './pages/knn/KNNClassification'
import KNNRegression from './pages/knn/KNNRegression'
import DTClassification from './pages/decision-tree/DTClassification'
import DTRegression from './pages/decision-tree/DTRegression'
import RFRegression from './pages/random-forest/RFRegression'
import RFClassification from './pages/random-forest/RFClassification'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<PageLayout />}>
          <Route path="/" element={<Home />} />
          <Route path="/linear/standard" element={<StandardLR />} />
          <Route path="/linear/lasso" element={<LassoLR />} />
          <Route path="/linear/ridge" element={<RidgeLR />} />
          <Route path="/linear/elasticnet" element={<ElasticNetLR />} />
          <Route path="/logistic" element={<LogisticReg />} />
          <Route path="/knn/classification" element={<KNNClassification />} />
          <Route path="/knn/regression" element={<KNNRegression />} />
          <Route path="/decision-tree/classification" element={<DTClassification />} />
          <Route path="/decision-tree/regression" element={<DTRegression />} />
          <Route path="/random-forest/regression" element={<RFRegression />} />
          <Route path="/random-forest/classification" element={<RFClassification />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
