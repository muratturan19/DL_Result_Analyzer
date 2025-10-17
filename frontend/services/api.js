const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error("Failed to fetch health status");
  }
  return response.json();
}

export async function fetchResults() {
  // Placeholder request for future implementation.
  return Promise.resolve([
    { id: 1, model: "ResNet", accuracy: 0.925, loss: 0.34 },
    { id: 2, model: "DenseNet", accuracy: 0.901, loss: 0.41 }
  ]);
}
