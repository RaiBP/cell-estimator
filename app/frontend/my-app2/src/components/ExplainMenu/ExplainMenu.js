import Button from '../Button/Button'
import './ExplainMenu.css'
import '../Menu/Menu.css'


function ExplainMenu({ handleOptionClick, selectorOptions, handleXAxisFeatureChange, handleYAxisFeatureChange}){
    
    return(
        <div className="explain-menu" style={{ left: '${x}px', top: '${y}px'}}>
            <Button className="explain-menu-button" onClick={() => handleOptionClick('close')}>Close</Button>
            <FeatureSelector className="explain-menu-selector-container" handleChange={handleXAxisFeatureChange} caption="Choose a feature for the X-axis" options={selectorOptions} />
            <FeatureSelector className="explain-menu-selector-container" handleChange={handleYAxisFeatureChange} caption="Choose a feature for the Y-axis" options={selectorOptions} />
            <Button className="explain-menu-button" onClick={() => handleOptionClick('plot')}>Plot</Button>
        </div>) 
}

function FeatureSelector({ handleChange, caption, options }) {
  return (
    <div className="explain-menu-selector-container">
      <label htmlFor='feature-selector' className="selector-label"> {caption} </label>
      <select id='feature-selector' className="selector" onChange={handleChange}>
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
