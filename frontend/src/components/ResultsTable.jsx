import React from "react";

const ResultsTable = () => {
  const placeholderRows = [
    { id: 1, model: "ResNet", accuracy: "92.5%", loss: "0.34" },
    { id: 2, model: "DenseNet", accuracy: "90.1%", loss: "0.41" }
  ];

  return (
    <table className="results-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Accuracy</th>
          <th>Loss</th>
        </tr>
      </thead>
      <tbody>
        {placeholderRows.map((row) => (
          <tr key={row.id}>
            <td>{row.model}</td>
            <td>{row.accuracy}</td>
            <td>{row.loss}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default ResultsTable;
