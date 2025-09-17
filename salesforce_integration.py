import requests
import logging
from typing import Dict, Any
import os

class SalesforceProductConnector:
    """Simple Salesforce connector for Product2 object only."""
    
    def __init__(self, instance_url: str, email: str, password: str, security_token: str):
        self.instance_url = instance_url
        self.email = email
        self.password = password
        self.security_token = security_token
        self.access_token = None
        self.api_version = 'v56.0'
        self.session = requests.Session()
        self.logger = logging.getLogger('SalesforceProductConnector')

    def authenticate(self) -> bool:
        """Authenticate with Salesforce using email/password/token and Connected App env credentials."""
        login_base = os.getenv('SF_LOGIN_URL', 'https://login.salesforce.com')
        auth_url = f"{login_base.rstrip('/')}/services/oauth2/token"

        client_id = os.getenv('SF_CLIENT_ID')
        client_secret = os.getenv('SF_CLIENT_SECRET')

        if not client_id or not client_secret:
            self.logger.error('❌ Missing SF_CLIENT_ID or SF_CLIENT_SECRET environment variables')
            return False

        auth_data = {
            'grant_type': 'password',
            'username': self.email,
            'password': self.password + self.security_token,
            'client_id': client_id,
            'client_secret': client_secret
        }
        
        try:
            response = self.session.post(auth_url, data=auth_data)
            
            if response.status_code == 200:
                auth_result = response.json()
                self.access_token = auth_result['access_token']
                self.instance_url = auth_result['instance_url']
                
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/json'
                })
                
                self.logger.info('✅ Authenticated with Salesforce successfully')
                return True
            else:
                error_msg = response.json() if response.content else response.text
                self.logger.error(f'❌ Authentication failed: {error_msg}')
                return False
                
        except Exception as e:
            self.logger.error(f'❌ Authentication error: {str(e)}')
            return False

    def upsert_product(self, sku: str, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🔧 FIXED: Upsert product excluding External ID field from payload.
        
        Args:
            sku: Product SKU (used as external ID in URL)
            product_data: Dictionary with product fields
            
        Returns:
            Dictionary with success status and details
        """
        if not self.access_token:
            if not self.authenticate():
                return {'success': False, 'error': 'Authentication failed'}

        # 🔧 CRITICAL FIX: Remove SKU__c from payload to avoid INVALID_FIELD error
        payload = {k: v for k, v in product_data.items() if k != 'SKU__c'}
        
        # External ID goes in URL only, not in payload
        url = f"{self.instance_url}/services/data/{self.api_version}/sobjects/Product2/SKU__c/{sku}"

        try:
            response = self.session.patch(url, json=payload)
            
            if response.status_code in (200, 201, 204):
                self.logger.info(f'✅ Product upserted successfully: {sku}')
                return {
                    'success': True, 
                    'sku': sku,
                    'operation': 'created' if response.status_code == 201 else 'updated'
                }
            else:
                error_detail = response.json() if response.content else response.text
                self.logger.error(f'❌ Failed to upsert product {sku}: {error_detail}')
                return {'success': False, 'error': error_detail}
                
        except Exception as e:
            self.logger.error(f'❌ Product upsert request failed: {str(e)}')
            return {'success': False, 'error': str(e)}

    def test_connection(self) -> bool:
        """Test Salesforce connection."""
        try:
            if not self.access_token:
                if not self.authenticate():
                    return False
            
            test_url = f"{self.instance_url}/services/data/{self.api_version}/query/"
            params = {'q': 'SELECT Id FROM Product2 LIMIT 1'}
            response = self.session.get(test_url, params=params)
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f'❌ Connection test failed: {str(e)}')
            return False

def map_document_to_product(document_data: Dict[str, Any], document_id: str) -> Dict[str, Any]:
    """
    🔧 FIXED: Map document data, SKU__c will be excluded from payload during upsert.
    """
    
    product_name = (
        document_data.get('product_name') or 
        document_data.get('item_name') or 
        document_data.get('description') or 
        f'Product from {document_id}'
    )
    
    # Include SKU__c here - it will be removed during upsert
    product_data = {
        'Name': str(product_name).strip()[:80],  # Required field
        'SKU__c': document_data.get('sku') or document_data.get('product_code') or document_id,
        'IsActive': True,
        'Description': str(document_data.get('description', ''))[:255] if document_data.get('description') else None
    }
    
    # Remove None values
    return {k: v for k, v in product_data.items() if v is not None}

def process_document_to_salesforce(document_data: Dict[str, Any], document_id: str, sf_connector: SalesforceProductConnector) -> Dict[str, Any]:
    """Process document and upsert to Salesforce Product2."""
    
    try:
        # Map document data to Product2 fields
        product_data = map_document_to_product(document_data, document_id)
        
        # Extract SKU for URL (will be removed from payload automatically)
        sku = product_data.get('SKU__c', document_id)
        
        # Upsert to Salesforce
        result = sf_connector.upsert_product(sku, product_data)
        
        if result['success']:
            return {
                'success': True,
                'document_id': document_id,
                'sku': sku,
                'operation': result.get('operation'),
                'salesforce_object': 'Product2'
            }
        else:
            return {
                'success': False,
                'document_id': document_id,
                'error': result.get('error')
            }
            
    except Exception as e:
        logging.error(f'❌ Error processing document {document_id}: {str(e)}')
        return {
            'success': False,
            'document_id': document_id,
            'error': str(e)
        }
