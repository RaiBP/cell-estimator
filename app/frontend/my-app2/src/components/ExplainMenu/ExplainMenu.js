
import Button from '../Button/Button'
import './ExplainMenu.css'
import '../Menu/Menu.css'


function ExplainMenu({ handleOptionClick, selectorOptions, handleXAxisFeatureChange, handleYAxisFeatureChange}){
    
    return(
        <div className="explain-menu" style={{ left: '${x}px', top: '${y}px'}}>
            <Button className="explain-menu-button" onClick={() => handleOptionClick('close')}>Close</Button>
            <FeatureSelector handleChange={handleXAxisFeatureChange} caption="Choose a feature for the X-axis" options={selectorOptions} />
            <FeatureSelector handleChange={handleYAxisFeatureChange} caption="Choose a feature for the Y-axis" options={selectorOptions} />
        </div>) 
}

function FeatureSelector({ handleChange, caption, options }) {
  return (
    <div className="selector-container">
      <label htmlFor='feature-selector' className="selector-label"> {caption} </label>
      <select id='dataset-selector' className="selector" onChange={handleChange}>
        {options.map((option, index) => (
          <option key={index} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  )
}

export { ExplainMenu }
